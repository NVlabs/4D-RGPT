"""STI-Bench task — prompt + scoring mirror MINT-SJTU's reference scripts.

Source: github.com/MINT-SJTU/STI-Bench  (opensource_test.py / openai_test.py)

Differences vs. our previous draft:
  - Prompt now opens with `From {time_start}s to {time_end}s.` and ends with
    `Please output only the option letter!` (canonical wording).
  - Candidate options are rendered as `(A) text` ... `(E) text`.
  - Answer extraction uses upstream's 6-pattern cascade rather than the
    generic prefix-then-letter sweep, so we score the same way as the paper.

Source data lives at STI_BENCH_ROOT/qa_processed.json (default `STI-Bench`).
Normalization caches a JSONL the HF `json` loader can ingest without schema
inference headaches.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


STI_BENCH_ROOT = os.environ.get("STI_BENCH_ROOT", "STI-Bench")
_NUM_OPTIONS = 5  # STI-Bench is fixed-arity 5-way MC (A-E)

# Verbatim from MINT-SJTU/STI-Bench opensource_test.py:
_ANSWER_PATTERNS = [
    r"\(([A-E])\)",                                # (A)
    r"Ans\s*=\s*['\"]?([A-E])['\"]?",              # Ans='C'
    r"Answer\s*[:=]\s*([A-E])",                    # Answer: B
    r"Option\s+([A-E])",                           # Option D
    r"\b([A-E])\s*(?:is|was)\s*correct",           # A is correct
    r"\b([A-E])[\.\)]\s*$",                        # C.  /  D)
]


def _normalize_cache_path() -> Path:
    return Path(STI_BENCH_ROOT) / ".cache" / "test_normalized.jsonl"


def _ensure_normalized() -> None:
    src = Path(STI_BENCH_ROOT) / "qa_processed.json"
    dst = _normalize_cache_path()
    if not src.exists():
        return
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src) as f:
        data = json.load(f)
    with open(dst, "w") as f:
        for item in data:
            item["id"] = str(item["id"])
            item["options"] = [str(o) for o in item.get("options", [])]
            item["answer"] = str(item.get("answer", ""))
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


_ensure_normalized()


def stibench_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    return [str(Path(STI_BENCH_ROOT) / "video" / doc["video"])]


def stibench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    # Upstream renders candidates as `(LETTER) text` keyed off a {letter:text}
    # dict; qa_processed.json gives us options as a parallel list, so we zip
    # against A..E in order.
    options = doc["options"]
    letters = [chr(ord("A") + i) for i in range(len(options))]
    cand_str = "\n".join(f"({L}) {opt}" for L, opt in zip(letters, options))
    ts, te = doc.get("time_start"), doc.get("time_end")
    return (
        f"From {ts} s to {te} s. {doc['question']}\n"
        f"{cand_str}\n"
        "Please output only the option letter!"
    )


def stibench_doc_to_target(doc: dict[str, Any]) -> str:
    return doc["answer"]


def _extract_answer(text: str) -> str:
    """Upstream's cascade — first matching pattern wins. Empty string if none."""
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Upstream falls back to checking the last character of the raw response —
    # see openai_test.py's `model_out[-1] in "ABCDE"`. Replicate that to keep
    # scoring parity for short responses like "C".
    stripped = text.strip()
    if stripped and stripped[-1] in "ABCDE":
        return stripped[-1]
    return ""


def stibench_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    pred_raw = results[0] if results else ""
    pred_letter = _extract_answer(pred_raw)
    gt_letter = doc["answer"]
    return {
        "stibench_accuracy": {
            "id": doc["id"],
            "scene": doc.get("scene", ""),
            "task": doc.get("task", ""),
            "pred_raw": pred_raw,
            "pred_letter": pred_letter,
            "gt_letter": gt_letter,
            "correct": 1.0 if (gt_letter and pred_letter == gt_letter) else 0.0,
        }
    }


def stibench_aggregate_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    by_task: defaultdict[str, list[float]] = defaultdict(list)
    by_scene: defaultdict[str, list[float]] = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r["correct"])
        by_scene[r["scene"]].append(r["correct"])

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    overall = _mean([r["correct"] for r in results])
    logger.info(f"STI-Bench overall: {overall:.1%} ({len(results)} items)")
    for task, xs in sorted(by_task.items()):
        logger.info(f"  task[{task}]: {_mean(xs):.1%} ({len(xs)})")
    for scene, xs in sorted(by_scene.items()):
        logger.info(f"  scene[{scene}]: {_mean(xs):.1%} ({len(xs)})")
    return overall
