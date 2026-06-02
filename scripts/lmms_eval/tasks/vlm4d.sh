#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# VLM4D eval (ShijieZhou-UCLA/VLM4D, ICCV 2025).
# Source data:  data/registry.yaml key `vlm4d` (commented out) → $VLM4D_ROOT
#               real_mc_processed.json + synthetic_mc_processed.json (merged
#               by our normalizer into one JSONL with a `split` field).
# Task config:  tools/vlm/lmms-eval/ext/tasks/vlm4d/
# Frames: 16 (paper setting). Upstream main.py defaults to 10 — see
# docs/lmms_eval_frame_counts.md for how to switch back if needed.
#
# NOTE: upstream uses an LLM-as-judge (o4-mini) over CoT responses for the
# published numbers; this task uses upstream's Direct-Output (DO) prompt
# with rule-based letter extraction so it runs offline.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

run_lmms_eval vlm4d "$@"
