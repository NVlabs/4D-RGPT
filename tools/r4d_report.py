#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Aggregate r4d_bench eval results across runs/checkpoints from BOTH pipelines:

  lmms_eval  →  runs/eval/<run>/lmms_eval/r4d_bench/model__checkpoint-<step>/*_results.json
  eval_old   →  runs/eval/<run>-step<step>/r4d/report.txt        (parsed Avg)
             →  runs/eval/<run>-step<step>/r4d_bench/report.txt  (same fallback)

Run from the repo root:

  # Default: all runs, side-by-side accuracy per checkpoint (markdown table)
  python tools/r4d_report.py

  # Restrict to specific runs (substring match against dir name)
  python tools/r4d_report.py --runs dev lora_v3

  # Plain ASCII instead of markdown
  python tools/r4d_report.py --format plain

The "eval_old" column is empty for any (run, step) that hasn't been evaluated
with the legacy pipeline yet — submit with `scripts/nvila/eval_old.sh`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

EVAL_ROOT = Path("runs/eval")
DEFAULT_TASK = "r4d_bench"
# eval_old uses the legacy task name `r4d`; lmms-eval uses `r4d_bench`.
EVAL_OLD_TASK = "r4d"


def lmms_eval_acc(run: str, step: int, task: str) -> float | None:
    """Latest `r4d_accuracy,none` from any *_results.json under the lmms_eval ckpt dir."""
    ckpt_dir = EVAL_ROOT / run / "lmms_eval" / task / f"model__checkpoint-{step}"
    files = sorted(glob.glob(str(ckpt_dir / "*_results.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    return data.get("results", {}).get(task, {}).get("r4d_accuracy,none")


# Match the 4th row of the eval_old report (the data row with `&` separators).
# Format: "  29.5  & 26.4     & 30.6      & 25.5 & ..."
# The Avg column comes first. eval_old appends every run, so we take the LAST
# matching row (latest eval) — earlier rows are from older eval invocations.
_EVAL_OLD_AVG = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*&", re.MULTILINE)


def eval_old_avg(run: str, step: int) -> float | None:
    """Latest Avg metric from the eval_old report.txt for this checkpoint.

    Falls back through both task-name conventions (`r4d` for the legacy
    datamodule registry, `r4d_bench` for the lmms-eval-aligned name).
    """
    for task in (EVAL_OLD_TASK, DEFAULT_TASK):
        report = EVAL_ROOT / f"{run}-step{step}" / task / "report.txt"
        if not report.exists():
            continue
        text = report.read_text()
        matches = _EVAL_OLD_AVG.findall(text)
        if matches:
            # Last entry wins (most recent append). eval_old emits Avg as a
            # percentage (e.g. "29.5") — already 0–100 scale, not 0–1.
            return float(matches[-1])
    return None


def discover_runs() -> list[str]:
    """All `runs/eval/<run>` dirs that contain at least one lmms_eval r4d_bench
    result OR an eval_old `<run>-step<N>/r4d{,_bench}` sibling.

    Returns canonical (non-step-suffixed) run names.
    """
    runs: set[str] = set()
    for d in EVAL_ROOT.glob("*"):
        if not d.is_dir():
            continue
        if (d / "lmms_eval" / DEFAULT_TASK).exists():
            runs.add(d.name)
            continue
        # eval_old path: <run>-step<N>/r4d{,_bench}/report.txt — strip the suffix
        m = re.match(r"(.+)-step\d+$", d.name)
        if m and ((d / EVAL_OLD_TASK / "report.txt").exists() or
                  (d / DEFAULT_TASK / "report.txt").exists()):
            runs.add(m.group(1))
    return sorted(runs)


def list_ckpt_steps(run: str, task: str) -> list[int]:
    steps: set[int] = set()
    # lmms_eval steps
    for d in glob.glob(str(EVAL_ROOT / run / "lmms_eval" / task / "model__checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        if m:
            steps.add(int(m.group(1)))
    # eval_old steps — sibling <run>-step<N> dirs
    for d in glob.glob(str(EVAL_ROOT / f"{run}-step*")):
        m = re.search(r"-step(\d+)$", d)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def build_table(runs: list[str], task: str) -> dict[str, dict[int, tuple[float | None, float | None]]]:
    """{run -> {step -> (lmms_acc, eval_old_avg)}}, both in 0–100 scale."""
    out: dict[str, dict[int, tuple]] = {}
    for run in runs:
        per_step: dict[int, tuple] = {}
        for step in list_ckpt_steps(run, task):
            lmms = lmms_eval_acc(run, step, task)
            if lmms is not None:
                lmms = lmms * 100.0  # 0–1 → 0–100
            old = eval_old_avg(run, step)  # already 0–100
            per_step[step] = (lmms, old)
        if per_step:
            out[run] = per_step
    return out


def fmt(v: float | None) -> str:
    return f"{v:>6.1f}%" if v is not None else "    -  "


def render_plain(table: dict[str, dict[int, tuple]]) -> str:
    lines: list[str] = []
    for run, per_step in table.items():
        lines.append(f"\n=== {run} ===")
        lines.append(f"{'step':>6} | {'lmms_eval':>9} | {'eval_old':>9}")
        lines.append("-" * 35)
        for step in sorted(per_step.keys()):
            lmms, old = per_step[step]
            lines.append(f"{step:>6} | {fmt(lmms):>9} | {fmt(old):>9}")
        # best-per-run summary
        best_lmms = max((v[0] for v in per_step.values() if v[0] is not None), default=None)
        best_old = max((v[1] for v in per_step.values() if v[1] is not None), default=None)
        lines.append(f"  best | {fmt(best_lmms):>9} | {fmt(best_old):>9}")
    return "\n".join(lines)


def render_markdown(table: dict[str, dict[int, tuple]]) -> str:
    lines: list[str] = []
    for run, per_step in table.items():
        lines.append(f"\n### {run}")
        lines.append("")
        lines.append("| step | lmms_eval | eval_old |")
        lines.append("|---|---|---|")
        for step in sorted(per_step.keys()):
            lmms, old = per_step[step]
            lines.append(f"| {step} | {fmt(lmms).strip()} | {fmt(old).strip()} |")
        best_lmms = max((v[0] for v in per_step.values() if v[0] is not None), default=None)
        best_old = max((v[1] for v in per_step.values() if v[1] is not None), default=None)
        lines.append(f"| **best** | **{fmt(best_lmms).strip()}** | **{fmt(best_old).strip()}** |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", help="Substring filter against run dir names. Default: all discovered runs.")
    parser.add_argument("--task", default=DEFAULT_TASK, help=f"lmms_eval task name (default: {DEFAULT_TASK})")
    parser.add_argument("--format", choices=("markdown", "plain"), default="markdown")
    args = parser.parse_args()

    discovered = discover_runs()
    if args.runs:
        runs = [r for r in discovered if any(filt in r for filt in args.runs)]
    else:
        runs = discovered

    if not runs:
        print(f"No runs found under {EVAL_ROOT} matching {args.runs or '*'}.", file=sys.stderr)
        return 1

    table = build_table(runs, args.task)
    if not table:
        print("No eval results found.", file=sys.stderr)
        return 1

    renderer = render_markdown if args.format == "markdown" else render_plain
    print(renderer(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
