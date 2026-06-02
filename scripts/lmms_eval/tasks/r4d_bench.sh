#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# R4D-Bench eval (CVPR'26 — 4D-RGPT: Toward Region-level 4D Understanding).
# Source data:  R4D-Bench/ (HF dataset nvidia/R4D-Bench; see R4D-Bench/README.md)
# Task config:  tools/vlm/lmms-eval/ext/tasks/r4d_bench/  (symlinked into the
#               lmms-eval submodule's tasks/ tree by tools/vlm/lmms-eval/setup.sh)
# Frames: 16 (paper setting, default for this repo). The original eval/qwen.py
# and eval/nvila.py runs used 32 — pass MAX_FRAMES=32 to reproduce those numbers.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

run_lmms_eval r4d_bench "$@"
