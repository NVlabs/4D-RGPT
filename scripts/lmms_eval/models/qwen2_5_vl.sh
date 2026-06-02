#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run a Qwen2.5-VL family model against one or more lmms-eval tasks.
#
# Usage:
#   bash scripts/lmms_eval/models/qwen2_5_vl.sh
#   bash scripts/lmms_eval/models/qwen2_5_vl.sh stibench vstibench
#   MODEL_PATH=inclusionAI/ViLaSR bash scripts/lmms_eval/models/qwen2_5_vl.sh

# Qwen2.5-VL needs no extra model args beyond pretrained + max_frames_num.
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-7B-Instruct"}
export MODEL_BACKEND="qwen2_5_vl"
export MODEL_ARGS_EXTRA=""

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
dispatch_tasks "$@"
