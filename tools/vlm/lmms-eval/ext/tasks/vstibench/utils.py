"""VSTI-Bench task — vendored from VLM-3R upstream.

Source: github.com/VITA-Group/VLM-3R / thinking-in-space/lmms_eval/tasks/vstibench/utils.py
Logic kept byte-for-byte so numbers compare apples-to-apples against the
published VLM-3R leaderboard. The only changes are thin renamings
(`vsibench_*` → `vstibench_*`) for consistency with the task name. To track
an upstream change, replace the function bodies — the surrounding constants
and aggregation pattern are stable.
"""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger


MCA_QUESTION_TYPES = [
    "obj_obj_relative_pos_nf",
    "obj_obj_relative_pos_ud",
    "obj_obj_relative_pos_lr",
    "camera_obj_rel_dist_v1",
    "camera_obj_rel_dist_v2",
    "camera_obj_rel_dist_v3",
    "camera_movement_direction",
]
NA_QUESTION_TYPES = [
    "camera_obj_abs_dist",
    "camera_displacement",
    "camera_obj_dist_change",
]

METRICS_FOR_MCA = {"accuracy": "exact_match"}
METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

WORST_CASE_FOR_METRICS = {"accuracy": 0.0, "MRA:.5:.95:.05": 0.0}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vstibench.yaml", "r") as f:
    safe_data = [l for l in f.readlines() if "!function" not in l]
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vstibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, doc["video_path"])
    if not os.path.exists(video_path):
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vstibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or \
            "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or \
            "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info("LMMS_EVAL_SHUFFLE_DOCS set — shuffling.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    try:
        return float(pred)
    except BaseException:
        return None


def vstibench_process_results(doc, results):
    doc["prediction"] = results[0]
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc["prediction"]), doc["mc_answer"])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(
                    to_float(fuzzy_matching(doc["prediction"])),
                    to_float(doc["ground_truth"]),
                )
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")
    return {"vstibench_score": doc}


def vstibench_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {}

    for question_type, idxs in results.groupby("question_type").groups.items():
        per_qt = results.iloc[idxs]
        metric_keys = METRICS_FOR_MCA.keys() if question_type in MCA_QUESTION_TYPES else METRICS_FOR_NA.keys()
        for metric in metric_keys:
            output[f"{question_type}_{metric}"] = per_qt[metric].mean()

    output["overall"] = sum(output.values()) / len(output)
    eval_logger.info(f"VSTI-Bench results: {output}")
    return output["overall"] * 100.0
