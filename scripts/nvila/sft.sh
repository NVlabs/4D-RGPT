# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

source .venv/bin/activate
source scripts/setups/env.sh

MODEL_PATH=${1:-"Efficient-Large-Model/NVILA-Lite-8B"}
MODEL_NAME=$(basename $MODEL_PATH)

DEFAULT_RUN_NAME=${2:-"$MODEL_NAME-4D-RGPT"}

DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1024
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2

DATASETS=(
    # --- Full train sets ---
    sat
    vstibench
    robofac
    wolf

)
if [ -z "$3" ] || [ "$3" = "-" ]; then
    DATA_MIXTURE=$(IFS=+; echo "${DATASETS[*]}")
else
    DATA_MIXTURE=$3
fi
NUM_TRAIN_EPOCHS=10


OUTPUT_DIR="runs/train/$DEFAULT_RUN_NAME"

source scripts/setups/train.sh

n_node=${SLURM_JOB_NUM_NODES:-1}

if [ "$NNODES" = "1" ]; then
    echo "Detected on single machine. Automatically set batch size to 2 for debugging."
    PER_DEVICE_TRAIN_BATCH_SIZE=2
    GRADIENT_ACCUMULATION_STEPS=16
    OUTPUT_DIR=runs/train/debug/$DEFAULT_RUN_NAME
fi

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m llava.train.train_mem \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $MODEL_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --region_extractor regiongpt \
        --low_level_perception l4p \
        --tune_vision_tower False \
        --tune_mm_projector False \
        --tune_language_model True \
        --tune_region_extractor True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 100 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 16384 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --num_video_frames 16 \
        --report_to none \
        --distill_tasks depth,flow_2d_backward,dyn_mask,camray \
        --ld_weight 0.01 \
        --ed_weights "depth=0.1,flow_2d_backward=0.001,dyn_mask=0.01,camray=0.00001" \
        "${@:4}"
