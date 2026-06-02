#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Submit a non-interactive multi-node training job. Wraps the provided
# command (e.g. `scripts/nvila/sft.sh ...`) inside the same pyxis container
# + startup logic that interactive.sh uses, so the runtime env matches.
#
# Usage:
#   bash scripts/slurm/submit.sh scripts/nvila/sft.sh [sft args...]
#
# Env overrides:
#   NUM_NODES (default 8), NUM_GPUS (default 8), DURATION (default 4 hours)

NUM_NODES=${NUM_NODES:-8}
NUM_GPUS=${NUM_GPUS:-8}
DURATION=${DURATION:-4}

if [ -f .env ]; then
    source .env
fi

if [ -z "${SLURM_ACCOUNT}" ]; then
    echo "Error: SLURM_ACCOUNT not specified" >&2
    exit 1
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/slurm/submit.sh <script> [args...]" >&2
    exit 1
fi

source scripts/slurm/_startup.sh
IMAGE=$(resolve_image)

CMD="bash $@"

# Job runs asynchronously after submit_job returns, so we can't trap-rm
# the startup file. Leaving it under $HOME is fine (a few KB).
STARTUP_FILE=$(mktemp "$HOME/4d_rgpt_startup.XXXXXX.sh")
write_startup_file "$STARTUP_FILE" exec "$CMD"

submit_job \
    --account "${SLURM_ACCOUNT}" \
    --name 4d_rgpt \
    --partition polar4,polar3,grizzly,polar \
    --nodes $NUM_NODES --gpu $NUM_GPUS \
    --image "$IMAGE" \
    --time $DURATION \
    --pre_timeout_signal 10 \
    --autoresume_uninstrumented \
    --autoresume_ignore_failure \
    --logroot ./logs \
    --command "bash $STARTUP_FILE"

echo "Job submitted to SLURM"
echo "> $CMD"
echo "Startup file: $STARTUP_FILE"
