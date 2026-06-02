"""R4D-Bench task callbacks for lmms-eval.

Data layout (resolved against R4D_BENCH_ROOT, default "R4D-Bench"):
    videos/<benchmark>/<basename>.mp4          # full source video
    clipped/<benchmark>/<basename>.mp4         # trimmed video starting at time_start
    images/<benchmark>/<id>.png                # set-of-mark image (first frame + bbox overlay)
    test.json                                  # 1419 MCQ entries

Each entry carries `video` (clipped if available, else videos/) and `source`
(always the full untrimmed videos/ path). Video selection tries `video` first,
falls back to `source` if missing on disk.

Sta/Dyn split is by benchmark: stibench (388) is Static, vlm4d (1031) is Dynamic.
Task categories (from actual test.json, not the README which mislabels a couple):
    Static  (stibench): VG, DM, SR, SA, DP
    Dynamic (vlm4d):    T, R, C, FP

The 12-column report (Avg|Sta|Dyn|VG|DM|SR|R|C|T|FP|SA|DP) is emitted by the
aggregator. It flattens all 9 categories regardless of which benchmark they
fall under, matching the existing eval/utils/report.py layout.

Question fields:
    question      — canonical SoM-keyed prompt with <obj_N> references that
                    line up with the bbox-overlay image in som_image.
    question_raw  — free-text variant with full object names (no <obj_N>).
                    Used as a fallback only.

test.json has mixed int/str/None types in `id`, `options`, and `answer`.
We normalize once into <R4D_BENCH_ROOT>/.cache/test_normalized.jsonl so the
HF datasets `json` loader can infer a single consistent schema.
"""
from __future__ import annotations

import json
import os
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


R4D_BENCH_ROOT = os.environ.get("R4D_BENCH_ROOT", "R4D-Bench")

TASK_TO_ABBREV: dict[str, str] = {
    "3D Video Grounding": "VG",
    "Dimensional Measurement": "DM",
    "Spatial Relation": "SR",
    "Translational": "T",
    "Rotational": "R",
    "Counting": "C",
    "False Positives": "FP",
    "Speed & Acceleration": "SA",
    "Displacement & Path Length": "DP",
}

# Sta/Dyn split in the legacy report.txt is by TASK CATEGORY, not by benchmark.
# (Note: this disagrees with the README's "Splits 388/1031" table, which is by
# benchmark. We follow the report.txt convention so numbers match prior runs.)
STATIC_TASKS = {"VG", "DM", "SR"}
DYNAMIC_TASKS = {"T", "R", "C", "FP", "SA", "DP"}

_ANSWER_PREFIXES = [
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
]


def _normalize_cache_path() -> Path:
    return Path(R4D_BENCH_ROOT) / ".cache" / "test_normalized.jsonl"


def _ensure_normalized() -> None:
    """Coerce mixed-type fields in test.json into a uniform JSONL cache.

    HF datasets' `json` loader does pyarrow schema inference on the first
    batch, which trips on `id`/`options`/`answer` having int+str+None values.
    Casting everything to string up front sidesteps it.
    """
    src = Path(R4D_BENCH_ROOT) / "test.json"
    dst = _normalize_cache_path()
    if not src.exists():
        return  # caller will hit a clear error during dataset load
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src) as f:
        data = json.load(f)
    with open(dst, "w") as f:
        for item in data:
            item["id"] = str(item["id"])
            item["options"] = ["" if o is None else str(o) for o in item["options"]]
            item["answer"] = "" if item["answer"] is None else str(item["answer"])
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# Run at import time (before lmms-eval's load_dataset). No-op if already up to date.
_ensure_normalized()


def _resolve_video_path(doc: dict[str, Any]) -> str:
    """`video` is the canonical clipped/full path. Fall back to `source` if missing.

    For trimmed entries, `video` is `clipped/<benchmark>/<basename>.mp4` and
    `source` is the full untrimmed `videos/<benchmark>/<basename>.mp4`. For
    untrimmed entries, the two are identical. The fallback covers users who
    haven't run `helpers.py --clip` to generate clipped/.
    """
    video_path = os.path.join(R4D_BENCH_ROOT, doc["video"])
    if os.path.exists(video_path):
        return video_path
    return os.path.join(R4D_BENCH_ROOT, doc["source"])


def r4d_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    """Return [som_image (if present), video]. Video is clipped-first."""
    paths: list[str] = []
    som_rel = doc.get("som_image")
    if som_rel:
        som_path = os.path.join(R4D_BENCH_ROOT, som_rel)
        if os.path.exists(som_path):
            paths.append(som_path)
    paths.append(_resolve_video_path(doc))
    return paths


_SOM_PREAMBLE = (
    "The first image is a set-of-mark annotation of a key frame from the video, "
    "with bounding boxes labeled <obj_1>, <obj_2>, etc. References to <obj_N> in "
    "the question correspond to those labeled regions."
)


def r4d_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    # `question` is the canonical SoM-keyed prompt (uses <obj_N> refs that
    # match the bbox-overlay image returned by r4d_doc_to_visual);
    # `question_raw` is the free-text fallback (full object names). Apply
    # the SoM preamble whenever a som_image is attached so the model knows
    # what the <obj_N> tokens point at.
    question = doc.get("question") or doc.get("question_raw") or ""
    has_som = bool(doc.get("som_image"))
    options = doc["options"]
    letters = list(string.ascii_uppercase[: len(options)])
    option_lines = "\n".join(f"({L}) {opt}" for L, opt in zip(letters, options))
    parts = []
    if has_som:
        parts.append(_SOM_PREAMBLE)
    parts.append(question)
    parts.append(f"Options:\n{option_lines}")
    parts.append("Answer with only the letter of the correct option.")
    prompt = "\n\n".join(parts)
    if lmms_eval_specific_kwargs and (post := lmms_eval_specific_kwargs.get("post_prompt")):
        prompt = f"{prompt}\n{post}"
    return prompt


def r4d_doc_to_target(doc: dict[str, Any]) -> str:
    """The expected output: the letter of the correct option, or '' if answer missing.

    Note: must test `answer is None` rather than `not answer`. For Counting
    tasks the correct answer can be the integer 0, which is falsy but valid.
    """
    options = doc["options"]
    answer = doc["answer"]
    if answer is None or answer == "" or answer not in options:
        return ""
    letters = list(string.ascii_uppercase[: len(options)])
    return letters[options.index(answer)]


def _extract_letter(text: str, num_options: int) -> str:
    """Pull A/B/C/D/E from a free-form model response.

    On no match, return a uniformly random choice from the available letters —
    matches the legacy 4d_rvila `parse_choice` behavior where unmatched outputs
    got ~1/N credit instead of automatic zero. Without this fallback, models
    that emit refusals or CoT-without-a-letter score artificially low compared
    to the old pipeline's reported numbers.
    """
    s = text.strip()
    for prefix in _ANSWER_PREFIXES:
        s = s.replace(prefix, "")
    valid = string.ascii_uppercase[:num_options]
    m = re.search(rf"\(([{valid}])\)", s)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{valid}])\b", s)
    if m:
        return m.group(1)
    return random.choice(list(valid))


def r4d_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    pred_raw = results[0] if results else ""
    options = doc["options"]
    gt_letter = r4d_doc_to_target(doc)
    pred_letter = _extract_letter(pred_raw, len(options))
    correct = 1.0 if (gt_letter and pred_letter == gt_letter) else 0.0
    task_abbrev = TASK_TO_ABBREV.get(doc["task"], doc["task"])
    if task_abbrev in STATIC_TASKS:
        split = "static"
    elif task_abbrev in DYNAMIC_TASKS:
        split = "dynamic"
    else:
        split = "unknown"
    sample = {
        "id": doc["id"],
        "benchmark": doc["benchmark"],
        "task": doc["task"],
        "task_abbrev": task_abbrev,
        "split": split,
        "pred_raw": pred_raw,
        "pred_letter": pred_letter,
        "gt_letter": gt_letter,
        "correct": correct,
    }
    # lmms-eval stores one aggregated float per metric name. To surface the
    # per-category breakdown in the result JSON (not just the loguru log),
    # we register one metric per column in the 12-column report and hand the
    # SAME per-sample dict to each — the aggregator below picks the slice it
    # cares about via `task_abbrev` / `split`.
    return {name: sample for name in _METRIC_NAMES}


# Canonical 12-column report header order; matches the format of the legacy
# eval/utils/report.py output in runs/eval/*/r4d_bench/report.txt.
_REPORT_HEADERS = ["Avg", "Sta", "Dyn", "VG", "DM", "SR", "R", "C", "T", "FP", "SA", "DP"]


def _compute_12col_row(results: list[dict[str, Any]]) -> dict[str, float]:
    """Return {header: accuracy_in_[0,1]} for the 12-column report."""
    by_split: defaultdict[str, list[float]] = defaultdict(list)
    by_task: defaultdict[str, list[float]] = defaultdict(list)
    for r in results:
        by_split[r["split"]].append(r["correct"])
        by_task[r["task_abbrev"]].append(r["correct"])

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    row = {
        "Avg": _mean([r["correct"] for r in results]),
        "Sta": _mean(by_split.get("static", [])),
        "Dyn": _mean(by_split.get("dynamic", [])),
    }
    for ab in ("VG", "DM", "SR", "R", "C", "T", "FP", "SA", "DP"):
        row[ab] = _mean(by_task.get(ab, []))
    return row


def format_12col_report(results: list[dict[str, Any]]) -> str:
    """Render the 12-column accuracy table in the legacy tabulate 'custom' format.

    Reuses eval.utils.report.Report so the output is byte-identical to the
    existing runs/eval/*/r4d_bench/report.txt layout.
    """
    # Local import: project root must be on sys.path when this runs.
    # We import lazily so unit tests that don't need the formatter still pass.
    from eval.utils.report import Report

    row = _compute_12col_row(results)
    report = Report(title="r4d_bench", headers=_REPORT_HEADERS, floatfmt=".1%")
    report.add_row([row[h] for h in _REPORT_HEADERS])
    return str(report)


def r4d_aggregate_results(results: list[dict[str, Any]]) -> float:
    """Return overall accuracy in [0, 1]. Side effects:

    * Log the 12-column table via loguru.
    * If R4D_REPORT_PATH is set, append the table to that file (matching the
      existing `with open(report.txt, 'a')` behavior in eval/nvila.py).
    """
    if not results:
        return 0.0
    table = format_12col_report(results)
    logger.info("\n" + table)
    report_path = os.environ.get("R4D_REPORT_PATH")
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(table)
            f.write("\n")
    return _compute_12col_row(results)["Avg"]


# --- per-category aggregators -------------------------------------------------
# Each returns accuracy in [0, 1] over the subset of samples whose `task_abbrev`
# (for the 9 task columns) or `split` (for static/dynamic) matches. Empty subset
# returns 0.0 — same convention as `_mean` in `_compute_12col_row`.

def _accuracy_for(results: list[dict[str, Any]], key: str, value: str) -> float:
    sub = [r["correct"] for r in results if r.get(key) == value]
    return sum(sub) / len(sub) if sub else 0.0


def r4d_static(results):    return _accuracy_for(results, "split", "static")
def r4d_dynamic(results):   return _accuracy_for(results, "split", "dynamic")
def r4d_VG(results):        return _accuracy_for(results, "task_abbrev", "VG")
def r4d_DM(results):        return _accuracy_for(results, "task_abbrev", "DM")
def r4d_SR(results):        return _accuracy_for(results, "task_abbrev", "SR")
def r4d_R(results):         return _accuracy_for(results, "task_abbrev", "R")
def r4d_C(results):         return _accuracy_for(results, "task_abbrev", "C")
def r4d_T(results):         return _accuracy_for(results, "task_abbrev", "T")
def r4d_FP(results):        return _accuracy_for(results, "task_abbrev", "FP")
def r4d_SA(results):        return _accuracy_for(results, "task_abbrev", "SA")
def r4d_DP(results):        return _accuracy_for(results, "task_abbrev", "DP")


# Names that `r4d_process_results` populates and `r4d_bench.yaml` declares as
# metrics — order mirrors the legacy report header for human readers.
_METRIC_NAMES = (
    "r4d_accuracy",
    "r4d_static",
    "r4d_dynamic",
    "r4d_VG", "r4d_DM", "r4d_SR",
    "r4d_R",  "r4d_C",  "r4d_T",
    "r4d_FP", "r4d_SA", "r4d_DP",
)
