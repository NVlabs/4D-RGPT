#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run an NVILA / VILA-family model against one or more lmms-eval tasks.
#
# Usage:
#   bash scripts/lmms_eval/models/nvila.sh                      # all tasks, default model
#   bash scripts/lmms_eval/models/nvila.sh stibench             # one task
#   bash scripts/lmms_eval/models/nvila.sh stibench vstibench   # subset
#
# Override the checkpoint via env:
#   MODEL_PATH=runs/train/<run-name>/model \
#     bash scripts/lmms_eval/models/nvila.sh

# Use the project's in-tree lmms-eval adapter (`nvila_native`), registered
# into MODEL_REGISTRY_V2 by the `llava.eval.lmms` wrapper module that _lib.sh
# launches in place of `-m lmms_eval`. It calls model.generate_content() —
# the high-level NVILA API — instead of the upstream `vila` adapter's
# old-LLaVA generate(images=...) call, which our newer `media`-dict-based
# generate() can't accept. Works for any NVILA-family checkpoint loadable
# via llava.load().
export MODEL_PATH=${MODEL_PATH:-"Efficient-Large-Model/NVILA-Lite-8B"}
export MODEL_BACKEND="nvila_native"

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
dispatch_tasks "$@"
