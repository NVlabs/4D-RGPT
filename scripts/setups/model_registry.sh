# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/bin/bash

MODEL_NAME=${1:-"NVILA-8B"}

case "$MODEL_NAME" in
    "SpatialReasoner")
        MODEL_PATH=ccvl/SpatialReasoner
        ;;
    "SpaceR")
        MODEL_PATH=RUBBISHLIKE/SpaceR
        ;;
    "VG-LLM")
        MODEL_PATH=zd11024/vgllm-qa-vggt-8b
        ;;
    "ViLaSR")
        MODEL_PATH=inclusionAI/ViLaSR
        ;;
    "Qwen3-VL-8B-Instruct")
        MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
        ;;
    "Qwen2.5-VL-7B-Instruct")
        MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
        ;;
    "NVILA-Lite-2B")
        MODEL_PATH=Efficient-Large-Model/NVILA-Lite-2B
        ;;
    "NVILA-8B")
        MODEL_PATH=Efficient-Large-Model/NVILA-8B
        ;;
    "NVILA-Lite-8B")
        MODEL_PATH=Efficient-Large-Model/NVILA-Lite-8B
        ;;
    "NVILA-8B-Video")
        MODEL_PATH=Efficient-Large-Model/NVILA-8B-Video
        ;;
    "NVILA-15B")
        MODEL_PATH=Efficient-Large-Model/NVILA-15B
        ;;
    "NVILA-Lite-15B")
        MODEL_PATH=Efficient-Large-Model/NVILA-Lite-15B
        ;;
    "NVILA-Lite-15B-Video")
        MODEL_PATH=Efficient-Large-Model/NVILA-Lite-15B-Video
        ;;
    "LongVILA-7B")
        MODEL_PATH=Efficient-Large-Model/qwen2-7b-longvila-256f
        ;;
    "LongVILA-1.5B")
        MODEL_PATH=Efficient-Large-Model/qwen2-1.5b-longvila-256f
        ;;
    *)
        echo "Warning: Unknown model name '$MODEL_NAME'" >&2
        MODEL_PATH=$MODEL_NAME
        # exit 1
        ;;
esac

export MODEL_PATH=$MODEL_PATH
# if [[ "$MODEL_PATH" == *"/lustre/"* ]]; then
#     A="${A}-internal"
# else
#     A="${A}.normal"
# fi
