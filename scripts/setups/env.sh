# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_IB_TIMEOUT=100

# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Load project .env so code paths that read os.environ directly (notably
# llava/data/builder.py: VILA_DATASETS, VILA_MIXTURES) see what's documented
# in .env. data/config.py uses python-dotenv on its own, so this is for the
# shell-env-only readers.
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Ensure third-party submodules are checked out. Currently:
#   - tools/vlm/lmms-eval/repo
#   - third_party/L4P  (perception teacher used by --low_level_perception l4p)
# Idempotent; no-op when already initialized.
if [ -f .gitmodules ] && [ -d .git ]; then
    git submodule update --init --recursive --quiet
fi