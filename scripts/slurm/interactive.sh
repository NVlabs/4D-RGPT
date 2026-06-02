# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


NUM_GPUS=${1:-8}
NUM_NODES=${2:-1}

if [ -f .env ]; then
    source .env
fi

if [ -z "${SLURM_ACCOUNT}" ]; then
    echo "Error: SLURM_ACCOUNT not specified" >&2
    exit 1
fi

source scripts/slurm/_startup.sh
IMAGE=$(resolve_image)

STARTUP_FILE=$(mktemp "$HOME/4d_rgpt_startup.XXXXXX.sh")
trap 'rm -f "$STARTUP_FILE"' EXIT
write_startup_file "$STARTUP_FILE" interactive

submit_job \
    --account "${SLURM_ACCOUNT}" \
    --name 4d_rgpt \
    --partition batch_singlenode,grizzly,polar,polar3,polar4,interactive \
    --gpu $NUM_GPUS --nodes $NUM_NODES -i \
    --image "$IMAGE" \
    --time 4 \
    --more_srun_args=--pty \
    --command "bash $STARTUP_FILE"
