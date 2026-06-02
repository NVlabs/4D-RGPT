"""SAT (Spatial Aptitude Test) task — prompt + scoring mirror arijitray1993/SAT.

Source: github.com/arijitray1993/SAT  (custom_datasets/dataloaders.py:
RealSATDynamic). Two upstream conventions to preserve:

1. **Natural-language MC prompt** (not letter-labelled):
   `"Answer in natural language. {Q} Answer the question using a single
    word or phrase. Choose between the following options: \"opt1\" or \"opt2\"."`

2. **Circular evaluation**: each item is queried twice with the option
   ordering swapped. The pair counts correct only if BOTH orderings are
   answered correctly. This is what `SATDynamicReal.json` does in upstream
   (line 619: appends entry with [3] then [2] flipped). We materialize the
   same expansion in the JSONL cache; the aggregator pairs queries by
   (item_id, ordering_id) to compute the pair-correctness rate.

Scoring uses fuzzy text-contains against the correct answer (case-insensitive,
trimmed). Upstream's eval also uses text matching — see eval_funcs around
the `gt_answer in pred.lower()` checks.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


SAT_ROOT = os.environ.get("SAT_ROOT", "SAT")


def _normalize_cache_path() -> Path:
    return Path(SAT_ROOT) / ".cache" / "test_circular.jsonl"


def _ensure_normalized() -> None:
    """Expand each SAT_test.json item into two queries with swapped option order."""
    src = Path(SAT_ROOT) / "SAT_test.json"
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
            answers = list(item.get("answers", []))
            correct = item.get("correct_answer", "")
            distractors = [a for a in answers if a != correct]
            # ordering 0: (correct, distractor*),  ordering 1: (distractor*, correct)
            for order_idx, ordered in enumerate([
                [correct, *distractors],
                [*distractors, correct],
            ]):
                f.write(json.dumps({
                    "id": str(item["id"]),
                    "order_idx": order_idx,
                    "question": item["question"],
                    "image": list(item["image"]),
                    "answers_ordered": ordered,
                    "correct_answer": correct,
                    "question_type": item.get("question_type", ""),
                }, ensure_ascii=False) + "\n")


_ensure_normalized()


def sat_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    return [str(Path(SAT_ROOT) / "images" / p) for p in doc["image"]]


def sat_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    # `format_prompts` in upstream renders options as `"opt1" or "opt2"`.
    answer_choices_format = " or ".join(f'"{ans}"' for ans in doc["answers_ordered"])
    return (
        f"Answer in natural language. {doc['question']} "
        f"Answer the question using a single word or phrase. "
        f"Choose between the following options: {answer_choices_format}."
    )


def sat_doc_to_target(doc: dict[str, Any]) -> str:
    return doc["correct_answer"]


def _matches(pred: str, target: str) -> bool:
    """Upstream's text-contains check, lower-cased and stripped."""
    if not target:
        return False
    return target.lower().strip() in pred.lower().strip()


def sat_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    pred_raw = results[0] if results else ""
    correct = doc["correct_answer"]
    # An answer is correct iff it surfaces the correct option text AND does
    # NOT also surface a distractor (otherwise the model named both).
    correct_match = _matches(pred_raw, correct)
    distractor_match = any(
        _matches(pred_raw, d) for d in doc["answers_ordered"] if d != correct
    )
    per_query_correct = 1.0 if (correct_match and not distractor_match) else 0.0
    return {
        "sat_accuracy": {
            "id": doc["id"],
            "order_idx": doc["order_idx"],
            "question_type": doc.get("question_type", ""),
            "pred_raw": pred_raw,
            "correct": per_query_correct,
        }
    }


def sat_aggregate_results(results: list[dict[str, Any]]) -> float:
    """Pair-up by (id, order_idx) and count an item correct only when both
    orderings hit. This is the upstream "circular eval" criterion."""
    if not results:
        return 0.0
    by_id: defaultdict[str, dict[int, float]] = defaultdict(dict)
    qtype_of: dict[str, str] = {}
    for r in results:
        by_id[r["id"]][r["order_idx"]] = r["correct"]
        qtype_of[r["id"]] = r["question_type"]

    pair_correct: list[float] = []
    per_qtype: defaultdict[str, list[float]] = defaultdict(list)
    for item_id, orders in by_id.items():
        both = orders.get(0, 0.0) and orders.get(1, 0.0)
        v = 1.0 if both else 0.0
        pair_correct.append(v)
        per_qtype[qtype_of[item_id]].append(v)

    overall = sum(pair_correct) / len(pair_correct)
    logger.info(f"SAT circular-eval overall: {overall:.1%} ({len(pair_correct)} item-pairs)")
    for qt, xs in sorted(per_qtype.items()):
        logger.info(f"  question_type[{qt}]: {sum(xs)/len(xs):.1%} ({len(xs)})")
    return overall
