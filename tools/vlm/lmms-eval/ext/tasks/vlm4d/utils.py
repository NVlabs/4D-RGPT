"""VLM4D task — prompt mirrors ShijieZhou-UCLA/VLM4D (direct-output mode).

Source: github.com/ShijieZhou-UCLA/VLM4D  (utils/constant.py)

  MULTI_CHOICE_DO_PROMPT = Template(\"\"\"
  Question: $question
  $optionized_str

  Do not generate any intermediate reasoning process. Answer directly with the
  option letter from the given choices.
  \"\"\")

  optionized_str is built by   "\n".join(f"{key}: {value}" for key,value in choices.items())

Upstream's `acc_evaluation.py` then runs an LLM-as-judge (o4-mini) over the
raw response to decide correctness; we approximate that with rule-based
letter extraction. The DO prompt elicits short letter responses ("A"/"B"),
so on the happy path the rule-based and LLM-judge scores agree closely.
"""
from __future__ import annotations

import json
import os
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


VLM4D_ROOT = os.environ.get("VLM4D_ROOT", "VLM4D")
_NUM_OPTIONS = 4  # VLM4D MC is 4-way A/B/C/D


def _normalize_cache_path() -> Path:
    return Path(VLM4D_ROOT) / ".cache" / "test_normalized.jsonl"


def _ensure_normalized() -> None:
    dst = _normalize_cache_path()
    sources = [
        ("real",      Path(VLM4D_ROOT) / "QA" / "real_mc_processed.json"),
        ("synthetic", Path(VLM4D_ROOT) / "QA" / "synthetic_mc_processed.json"),
    ]
    existing = [s for _, s in sources if s.exists()]
    if not existing:
        return
    if dst.exists() and dst.stat().st_mtime >= max(s.stat().st_mtime for s in existing):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for split_name, src in sources:
            if not src.exists():
                continue
            with open(src) as g:
                data = json.load(g)
            for item in data:
                item["id"] = str(item["id"])
                item["split"] = split_name
                item["choices"] = {k: str(v) for k, v in item["choices"].items()}
                item["answer"] = str(item.get("answer", ""))
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


_ensure_normalized()


def vlm4d_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    return [str(Path(VLM4D_ROOT) / doc["video"])]


def vlm4d_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    # Upstream renders options as `A: text` with colon, no parentheses —
    # mirrors `[f"{k}: {v}" for k, v in choices.items()]`.
    optionized_str = "\n".join(f"{k}: {v}" for k, v in doc["choices"].items())
    return (
        f"Question: {doc['question']}\n"
        f"{optionized_str}\n\n"
        "Do not generate any intermediate reasoning process. "
        "Answer directly with the option letter from the given choices."
    )


def vlm4d_doc_to_target(doc: dict[str, Any]) -> str:
    """Resolve the answer TEXT (upstream stores text, not letter) to its letter."""
    answer_text = doc["answer"]
    for letter, text in doc["choices"].items():
        if text == answer_text:
            return letter
    return ""


def _extract_letter(text: str, num_options: int) -> str:
    s = text.strip()
    valid = string.ascii_uppercase[:num_options]
    # Bare leading letter "A" or "A:" or "A." — most common with DO prompt.
    m = re.match(rf"^\s*([{valid}])\b", s)
    if m:
        return m.group(1)
    # "Answer: A" / "The answer is A" — defensive against minor template drift.
    m = re.search(rf"answer\s*(?:is)?\s*[:\-]?\s*([{valid}])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Any standalone letter in the response.
    m = re.search(rf"\b([{valid}])\b", s)
    if m:
        return m.group(1)
    return ""


def vlm4d_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    pred_raw = results[0] if results else ""
    pred_letter = _extract_letter(pred_raw, _NUM_OPTIONS)
    gt_letter = vlm4d_doc_to_target(doc)
    return {
        "vlm4d_accuracy": {
            "id": doc["id"],
            "split": doc.get("split", ""),
            "question_type": doc.get("question_type", ""),
            "pred_raw": pred_raw,
            "pred_letter": pred_letter,
            "gt_letter": gt_letter,
            "correct": 1.0 if (gt_letter and pred_letter == gt_letter) else 0.0,
        }
    }


def vlm4d_aggregate_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    by_split: defaultdict[str, list[float]] = defaultdict(list)
    by_type: defaultdict[str, list[float]] = defaultdict(list)
    for r in results:
        by_split[r["split"]].append(r["correct"])
        by_type[r["question_type"]].append(r["correct"])

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    overall = _mean([r["correct"] for r in results])
    logger.info(f"VLM4D overall: {overall:.1%} ({len(results)} items)")
    for split, xs in sorted(by_split.items()):
        logger.info(f"  split[{split}]: {_mean(xs):.1%} ({len(xs)})")
    for qt, xs in sorted(by_type.items()):
        logger.info(f"  question_type[{qt}]: {_mean(xs):.1%} ({len(xs)})")
    return overall
