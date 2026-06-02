#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Top-level lmms-eval entry point. Mirrors scripts/nvila/eval.sh (which runs
# the legacy in-repo custom eval via eval/nvila.py) but routes the same task
# names through the lmms-eval framework. The two scripts share a CLI shape
# for A/B comparison against the same checkpoint.
#
# Usage:
#   bash scripts/lmms_eval.sh                                # default NVILA-Lite-8B, r4d_bench
#   bash scripts/lmms_eval.sh <ckpt_path>
#   bash scripts/lmms_eval.sh <ckpt_path> <task1,task2,...>
#
# Env overrides (read by scripts/lmms_eval/_lib.sh):
#   MODEL_BACKEND (default nvila_native), MAX_FRAMES (16), MODEL_NAME (auto),
#   OUTPUT_ROOT (runs/eval/<MODEL_NAME>/lmms_eval), MODEL_ARGS_EXTRA
#
# vs scripts/nvila/eval.sh:
#   - nvila/eval.sh   → torchrun eval/nvila.py        (legacy datamodule-based)
#   - lmms_eval.sh    → accelerate launch lmms-eval   (framework-based, used
#                                                      for paper-comparable
#                                                      and external models)

MODEL_PATH=${1:-"Efficient-Large-Model/NVILA-Lite-8B"}
TASKS=${2:-"r4d_bench"}

# Comma-separated task names → positional args for dispatch_tasks().
TASKS_ARGS="${TASKS//,/ }"

export MODEL_PATH
export MODEL_BACKEND="${MODEL_BACKEND:-nvila_native}"

# Reuse the existing per-task plumbing (cache symlinks, INCLUDE_PATH, model_args
# resolution, the project's `llava.eval.lmms` wrapper module that registers the
# nvila_native backend). dispatch_tasks iterates the requested tasks and tolerates
# per-task failures so a flaky benchmark doesn't kill the others.
source "$(dirname "${BASH_SOURCE[0]}")/lmms_eval/_lib.sh"
dispatch_tasks $TASKS_ARGS
