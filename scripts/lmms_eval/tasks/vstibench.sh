#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# VSTI-Bench eval (VITA-Group/VLM-3R).
# Source data:  pulled from HF Hub (Journey9ni/vstibench) → cached under $HF_HOME
# Task config:  tools/vlm/lmms-eval/ext/tasks/vstibench/
# Frames: 16 (paper setting). Upstream eval_vlm_3r_vstibench.sh uses 32 — see
# docs/lmms_eval_frame_counts.md for how to switch back if needed.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

run_lmms_eval vstibench "$@"
