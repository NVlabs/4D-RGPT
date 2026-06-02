# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Thin wrapper around lmms-eval's CLI that registers our in-tree model
# adapters into MODEL_REGISTRY_V2 before delegating. Lets us drop the
# deprecated LMMS_EVAL_PLUGINS env-var hook without making 4d_rgpt a
# pip-installable package (which would be needed for the entry-points
# path lmms-eval prefers).
#
# Usage (from scripts/lmms_eval/_lib.sh):
#     accelerate launch ... -m llava.eval.lmms -- ...lmms-eval args...
import sys

from lmms_eval.models import MODEL_REGISTRY_V2
from lmms_eval.models.registry_v2 import ModelManifest

MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="nvila_native",
        simple_class_path="llava.eval.lmms.models.nvila_native.NVILANative",
    ),
    overwrite=True,
)


# ----- Format accuracy-style metrics as xx.x% in lmms-eval's results table ---
#
# Upstream `make_table` renders metric values via `"%.4f" % v` when `v` is a
# float. Pre-mutating the result_dict to convert accuracy-style floats (those
# whose `higher_is_better=True` AND value lies in [0, 1]) to "xx.x%" strings
# makes the check fall through (string stays as-is), so the table prints
# percentages without us having to fork upstream code.
#
# Side effect: `v_numeric` (used only for the optional --baseline diff column)
# becomes None for these rows, so that column shows N/A. We don't use it.
#
# IMPORTANT: results.json is saved BEFORE this runs (inside cli_evaluate_single,
# via save_results_aggregated), so mutating the dict here only affects display.
import lmms_eval.utils as _lmms_utils
import lmms_eval.__main__ as _lmms_main

_orig_make_table = _lmms_utils.make_table

_AUX_SUFFIXES = (
    "_stderr", "_stderr_clt", "_stderr_clustered",
    "_expected_accuracy", "_consensus_accuracy",
    "_internal_variance", "_consistency_rate",
)


def _format_accuracy_floats_inplace(result_dict, column: str) -> None:
    target = result_dict.get(column, {})
    hib_all = result_dict.get("higher_is_better", {})
    for task_name, metrics in target.items():
        hib_for_task = hib_all.get(task_name, {}) or {}
        for key in list(metrics.keys()):
            metric, _, _ = key.partition(",")
            if metric.endswith(_AUX_SUFFIXES):  # leave stderr etc. as floats
                continue
            if metric.startswith("paired_") or metric == "alias":
                continue
            v = metrics[key]
            if not isinstance(v, float) or not hib_for_task.get(metric):
                continue
            if not (0.0 <= v <= 1.0):
                continue
            metrics[key] = f"{v * 100:.1f}%"


def _make_table_pct(result_dict, column: str = "results", sort_results: bool = False):
    _format_accuracy_floats_inplace(result_dict, column)
    return _orig_make_table(result_dict, column, sort_results)


_lmms_utils.make_table = _make_table_pct
# __main__.py did `from lmms_eval.utils import make_table` at import time, so
# patch its namespace too — the bound reference there is independent.
_lmms_main.make_table = _make_table_pct


# ----- Also save the table as Markdown alongside the JSON results ----------
#
# lmms-eval only writes `<date_id>_results.json` by default. Mirror it as
# `<date_id>_results.md` (the same table the CLI prints, in xx.x% form) so
# users can browse results without parsing JSON or re-running.
from pathlib import Path as _Path
import lmms_eval.loggers.evaluation_tracker as _ev_tracker

_orig_save_results_aggregated = _ev_tracker.EvaluationTracker.save_results_aggregated


def _save_results_aggregated_with_md(self, results, samples, datetime_str):
    _orig_save_results_aggregated(self, results, samples, datetime_str)
    if not self.output_path:
        return
    try:
        path = _Path(self.output_path).joinpath(self.general_config_tracker.model_name_sanitized)
        md_path = path.joinpath(f"{self.date_id}_results.md")
        md = _make_table_pct(results, "results")
        if "groups" in results:
            md += "\n" + _make_table_pct(results, "groups")
        md_path.write_text(md, encoding="utf-8")
    except Exception as exc:  # don't let markdown save kill a successful eval
        import loguru
        loguru.logger.warning(f"Failed to save markdown results: {exc}")


_ev_tracker.EvaluationTracker.save_results_aggregated = _save_results_aggregated_with_md


# ----- Conditionally re-format omnispatial's INFO-log accuracies as xx.x% ----
#
# Upstream `lmms_eval.tasks.omnispatial.utils.omnispatial_aggregate_results`
# logs Total Samples / Total Correct / per-sub-task / per-task accuracies as
# "0.xxxx" floats. We re-format to xx.x% — but only when the user is actually
# running omnispatial, because importing that module triggers a heavy
# `snapshot_download` at module top level.
def _patch_omnispatial_logger():
    import lmms_eval.tasks.omnispatial.utils as _omni
    from collections import defaultdict
    from loguru import logger as eval_logger

    def omnispatial_aggregate_results_pct(results):
        sub_task_to_scores = defaultdict(list)
        task_to_scores = defaultdict(list)
        total_samples = len(results)
        total_correct = 0
        for sample in results:
            score = 1 if sample["is_correct"] else 0
            total_correct += score
            sub_task_to_scores[sample["sub_task"]].append(score)
            task_to_scores[sample["task"]].append(score)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        sub_task_acc = {k: sum(v) / len(v) for k, v in sub_task_to_scores.items()}
        task_acc = {k: sum(v) / len(v) for k, v in task_to_scores.items()}

        eval_logger.info(f"{'Total Samples':<20}: {total_samples}")
        eval_logger.info(f"{'Total Correct':<20}: {total_correct}")
        eval_logger.info(f"{'Overall Accuracy':<20}: {accuracy * 100:.1f}%")
        eval_logger.info("")
        eval_logger.info(f"{'Per-Sub-Task Accuracy':<40}")
        eval_logger.info("-" * 40)
        for k, v in sub_task_acc.items():
            eval_logger.info(f"{k:<32}: {v * 100:.1f}%")
        eval_logger.info("=" * 40)
        eval_logger.info(f"{'Per-Task Accuracy':<40}")
        eval_logger.info("-" * 40)
        for k, v in task_acc.items():
            eval_logger.info(f"{k:<32}: {v * 100:.1f}%")
        eval_logger.info("=" * 40)
        return accuracy

    _omni.omnispatial_aggregate_results = omnispatial_aggregate_results_pct


# Only trigger the upstream import if the user actually asked for omnispatial.
# YAML resolution happens later in cli_evaluate via `!function utils.X` which
# does `getattr(module, name)` — patching the module attr before then means
# our function gets bound into the task config.
if any("omnispatial" in arg for arg in sys.argv):
    _patch_omnispatial_logger()


# ----- qwen-vl-utils: clamp `smart_nframes` for degenerate videos ----------
#
# Some R4D-Bench clips are trimmed with time_start ~= time_end and contain a
# single frame. qwen-vl-utils's smart_nframes raises
#   `nframes should in interval [2, 1], but got 16`
# in that case, then falls back to torchvision.io.read_video (which we've
# disabled because torchvision>=0.27 removed that API). Net effect: one bad
# video kills the whole task.
#
# Clamp nframes to [1, total_frames] when the original check would fail,
# so the offending sample completes (as effectively a single-frame image).
def _patch_qwenvl_smart_nframes():
    import qwen_vl_utils.vision_process as _qvl
    from loguru import logger as _qlog

    _orig = _qvl.smart_nframes

    def smart_nframes_safe(ele, total_frames, video_fps):
        try:
            return _orig(ele, total_frames=total_frames, video_fps=video_fps)
        except ValueError as exc:
            if "nframes should in interval" not in str(exc):
                raise
            if total_frames < _qvl.FRAME_FACTOR:
                clamped = max(1, total_frames)
            else:
                requested = ele.get("nframes", _qvl.FRAME_FACTOR)
                clamped = max(_qvl.FRAME_FACTOR, min(requested, _qvl.floor_by_factor(total_frames, _qvl.FRAME_FACTOR)))
            _qlog.warning(
                f"smart_nframes: clamped to {clamped} (video has {total_frames} frames, requested via {ele})"
            )
            return clamped

    _qvl.smart_nframes = smart_nframes_safe


try:
    _patch_qwenvl_smart_nframes()
except ImportError:
    pass  # qwen-vl-utils not installed for this run (e.g. NVILA-only).


from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
