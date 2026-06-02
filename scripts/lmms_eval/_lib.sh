#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Shared bootstrap for the lmms-eval shell scripts under tasks/ and models/.
# Not meant to be run directly — source it.
#
# Layout:
#   scripts/lmms_eval/_lib.sh         (this file)
#   scripts/lmms_eval/tasks/<task>.sh (sources _lib.sh, runs one task)
#   scripts/lmms_eval/models/<m>.sh   (sets backend, dispatches to tasks/*.sh)

# -u (nounset) breaks scripts/setups/env.sh which references unset vars
# (PYTHONPATH, CONDA_PREFIX, ...). Stick with errexit + pipefail.
set -eo pipefail

# Resolve the project root from this file's location so scripts can be
# invoked from any CWD (CI, sweep configs, slurm sbatch ...).
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_LIB_DIR/../.." && pwd)"
cd "$_REPO_ROOT"

# Project env (.venv activation, PYTHONPATH, NCCL flags, submodule init).
#
# LMMS_EVAL_VENV (default `.venv`) lets the matrix submitter (submit_eval_matrix.sh)
# swap in `.venv-v5-backup` for the transformers-5 mode. Bypass `activate` and
# prepend PATH manually: the v5 backup venv's activate script has a stale
# VIRTUAL_ENV pointer (never rewritten after the copy from .venv), so sourcing
# it silently falls back to the main .venv's transformers 4.46.
LMMS_EVAL_VENV=${LMMS_EVAL_VENV:-".venv"}
if [ ! -x "$LMMS_EVAL_VENV/bin/python" ]; then
    echo "Error: $LMMS_EVAL_VENV/bin/python not found. Override with LMMS_EVAL_VENV=<path>." >&2
    exit 1
fi
export VIRTUAL_ENV="$(realpath "$LMMS_EVAL_VENV")"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
source scripts/setups/env.sh

# Default arguments. Per-task scripts may override before invoking.
MODEL_PATH=${MODEL_PATH:-${1:-"Qwen/Qwen2.5-VL-7B-Instruct"}}
MODEL_BACKEND=${MODEL_BACKEND:-"qwen2_5_vl"}

# If MODEL_PATH is a local directory, make it absolute so it survives the
# chdir into LMMS_EVAL_CACHE_ROOT inside run_lmms_eval. HF hub identifiers
# (e.g. "Efficient-Large-Model/NVILA-Lite-8B") are left untouched.
if [ -d "$MODEL_PATH" ]; then
    MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
fi

# Derive a friendly run name for output paths / log suffixes:
#   runs/train/<run>/model/checkpoint-N → <run>     (project training convention)
#   else                                  → basename
# Override with MODEL_NAME=... if neither rule fits.
if [ -z "$MODEL_NAME" ]; then
    if [[ "$MODEL_PATH" == */runs/train/*/* ]]; then
        MODEL_NAME=$(echo "$MODEL_PATH" | sed -E 's@.*/runs/train/([^/]+)/.*@\1@')
    else
        MODEL_NAME=$(basename "$MODEL_PATH")
    fi
fi

# Frame sampling — fixed at 16 to match the 4D-RGPT paper / training setup
# (--num_video_frames 16) and the L4P encoder's 16-frame window. The original
# upstream-matching per-task counts (stibench=30, vstibench=32, vlm4d=10) are
# documented in docs/lmms_eval_frame_counts.md and can be re-applied via
# `MAX_FRAMES=<n> bash scripts/lmms_eval/<task>.sh ...`.
MAX_FRAMES=${MAX_FRAMES:-16}

# Where lmms-eval writes per-sample logs + summary JSON. Absolute so it
# doesn't depend on the lmms-eval launch CWD (we chdir into the dataset
# cache root just before launching).
OUTPUT_ROOT=${OUTPUT_ROOT:-"$_REPO_ROOT/runs/eval/$MODEL_NAME/lmms_eval"}
# Anchor OUTPUT_ROOT to absolute. We chdir into LMMS_EVAL_CACHE_ROOT just
# before launching lmms-eval, so a relative OUTPUT_ROOT (e.g. set by the
# matrix submitter) would resolve under the cache root and the results
# files end up silently misplaced. The framework still LOGS "Saving samples
# to <relative>" — looks fine, but the file lands elsewhere.
case "$OUTPUT_ROOT" in
    /*) ;;  # already absolute
    *)  OUTPUT_ROOT="$_REPO_ROOT/$OUTPUT_ROOT" ;;
esac
mkdir -p "$OUTPUT_ROOT"

# Single designated location for ALL lmms-eval dataset caches and symlinks,
# so the repo root stays clean (no stray STI-Bench/, omnispatial/, etc.).
# We chdir here right before `accelerate launch`, which makes both our
# project-relative `data_files` paths AND upstream task YAMLs' relative
# `cache_dir:` settings (e.g. omnispatial's `cache_dir: omnispatial`,
# vstibench's `cache_dir: vstibench`) resolve inside this one folder.
LMMS_EVAL_CACHE_ROOT=${LMMS_EVAL_CACHE_ROOT:-"$_REPO_ROOT/.cache/lmms_eval_datasets"}
mkdir -p "$LMMS_EVAL_CACHE_ROOT"

# Our task YAMLs reference data files via project-relative paths like
# `STI-Bench/.cache/test_normalized.jsonl` (HF datasets resolves this
# against CWD). Symlink each dataset name inside LMMS_EVAL_CACHE_ROOT.
# If a stale link points elsewhere (e.g. an older target), repoint it.
_DATASETS_BASE=/lustre/fs12/portfolios/nvr/projects/nvr_taiwan_rvos/datasets
_link_dataset() {
    local name="$1"; local target="$2"
    local link="$LMMS_EVAL_CACHE_ROOT/$name"
    if [ -L "$link" ] && [ "$(readlink "$link")" != "$target" ]; then
        rm "$link"
    fi
    if [ ! -e "$link" ] && [ -d "$target" ]; then
        ln -s "$target" "$link"
    fi
}
_link_dataset STI-Bench  "$_DATASETS_BASE/STI-Bench"
_link_dataset VSTI-Bench "$_DATASETS_BASE/VSTI-Bench"
_link_dataset SAT        "$_DATASETS_BASE/SAT"
_link_dataset VLM4D      "$_DATASETS_BASE/VLM4D"
# R4D-Bench is canonical in-repo (committed alongside the task code). The
# lustre copy uses an older `question_new`/`question_old` schema; the repo
# uses `question`/`question_raw`, which r4d_bench/utils.py keys off.
_link_dataset R4D-Bench  "$_REPO_ROOT/R4D-Bench"

# Our project-local task extensions (r4d_bench, stibench, vstibench, sat, vlm4d).
# Absolute so it survives the chdir into LMMS_EVAL_CACHE_ROOT below.
# Upstream-bundled tasks (mmsibench, omnispatial) don't need this path, but
# passing it is harmless — lmms-eval merges both sources.
INCLUDE_PATH=${INCLUDE_PATH:-"$_REPO_ROOT/tools/vlm/lmms-eval/ext/tasks"}

# Resolve number of GPUs.
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "${NPROC_PER_NODE}" -eq 0 ]; then NPROC_PER_NODE=1; fi

# Backend-specific extras appended to --model_args. Backends differ in what
# they accept here:
#   - qwen2_5_vl: just `pretrained=...,max_frames_num=...`
#   - nvila_native (this project's adapter): `model_path=...,num_video_frames=...`
# Set MODEL_ARGS_EXTRA per-call to thread anything else (LoRA paths, device
# maps, etc.).
MODEL_ARGS_EXTRA=${MODEL_ARGS_EXTRA:-""}

# Default task list, used by dispatch_tasks when no args are given.
ALL_TASKS=( r4d_bench stibench vstibench sat vlm4d mmsibench omnispatial )

# Invoked by tasks/<task>.sh to run lmms-eval for ONE task using the
# currently-set MODEL_PATH / MODEL_BACKEND / MODEL_ARGS_EXTRA / MAX_FRAMES.
# Args after the task name are appended verbatim to lmms-eval.
run_lmms_eval() {
    local task="$1"; shift
    local out_dir="$OUTPUT_ROOT/$task"
    mkdir -p "$out_dir"

    # Per-backend --model_args naming differs (kwarg names hit the backend
    # __init__ directly). Keep that knowledge here so per-task scripts and
    # per-model scripts stay simple.
    local model_args
    case "$MODEL_BACKEND" in
        nvila_native)
            model_args="model_path=$MODEL_PATH,num_video_frames=$MAX_FRAMES,conv_mode=auto"
            ;;
        qwen2_5_vl)
            # Qwen2_5_VL.__init__ uses `max_num_frames` (different from the
            # legacy `max_frames_num` other backends accept) and asserts no
            # unknown kwargs, so we have to spell it correctly.
            # flash_attention_2 keeps long-video evals (16 frames × ~600K
            # pixels + 8192-token max output) under an 80GB GPU's memory.
            model_args="pretrained=$MODEL_PATH,max_num_frames=$MAX_FRAMES,attn_implementation=flash_attention_2"
            ;;
        gpt4v)
            # API backend (OpenAI + Azure modes). `model_version` is the SKU
            # string (e.g. gpt-5-20250807) — NOT a HF path. For NVIDIA's
            # Azure-shape proxy, set API_TYPE=azure + AZURE_* env vars in the
            # caller (see scripts/lmms_eval/models/gpt5.sh).
            model_args="model_version=$MODEL_PATH,max_frames_num=$MAX_FRAMES"
            ;;
        *)
            model_args="pretrained=$MODEL_PATH,max_frames_num=$MAX_FRAMES"
            ;;
    esac
    if [ -n "$MODEL_ARGS_EXTRA" ]; then
        model_args="${model_args},${MODEL_ARGS_EXTRA}"
    fi

    # Run from the dataset cache root so all relative `cache_dir:` /
    # `data_files:` paths in task YAMLs (ours + upstream) land there.
    # `lmms_eval` is installed editable so it imports fine, but `llava/`
    # is not — env.sh only adds `.` to PYTHONPATH, which would resolve to
    # the cache dir after chdir. Re-anchor to the repo root explicitly.
    # `python -m accelerate.commands.launch` instead of `accelerate launch`:
    # the v5 backup venv's bin/accelerate has a hardcoded shebang pointing at
    # the main .venv (the copy never rewrote entry-point shebangs), so the
    # `accelerate` binary on PATH runs under transformers 4.46 regardless of
    # which venv we activated. `python` is a real symlink — PATH-resolved
    # correctly — so going through `-m` lands on the right interpreter.
    ( cd "$LMMS_EVAL_CACHE_ROOT" && \
      PYTHONPATH="$_REPO_ROOT:$PYTHONPATH" \
      python -m accelerate.commands.launch \
          --num_processes="$NPROC_PER_NODE" \
          -m llava.eval.lmms \
          --model "$MODEL_BACKEND" \
          --model_args "$model_args" \
          --tasks "$task" \
          --batch_size 1 \
          --log_samples \
          --log_samples_suffix "$MODEL_NAME" \
          --output_path "$out_dir" \
          --include_path "$INCLUDE_PATH" \
          "$@"
    )
}

# Invoked by models/<m>.sh after it has exported MODEL_PATH / MODEL_BACKEND /
# MODEL_ARGS_EXTRA. With no args, runs every task in ALL_TASKS; otherwise
# runs only the named tasks. Failures in one task don't abort the rest.
dispatch_tasks() {
    local tasks=( "$@" )
    if [ "${#tasks[@]}" -eq 0 ]; then
        tasks=( "${ALL_TASKS[@]}" )
    fi

    local tasks_dir="$_LIB_DIR/tasks"
    for t in "${tasks[@]}"; do
        local script="$tasks_dir/${t}.sh"
        if [ ! -f "$script" ]; then
            echo "  [SKIP] no task script $script"
            continue
        fi
        echo "================================================================"
        echo " $t  (model=$MODEL_PATH, backend=$MODEL_BACKEND)"
        echo "================================================================"
        if ! bash "$script"; then
            echo "  [FAIL] $t — continuing with remaining tasks"
        fi
    done

    echo "Done. Logs under $OUTPUT_ROOT/"
}
