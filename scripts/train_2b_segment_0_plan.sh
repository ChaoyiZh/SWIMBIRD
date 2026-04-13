#!/bin/bash

# export WANDB_DISABLED=true
export WANDB_DISABLED=false
export WANDB_PROJECT="${WANDB_PROJECT:-SwimBird}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME:-swimbird_singlenode_2b_segment_0_plan}}"
export WANDB_WATCH="${WANDB_WATCH:-false}"
export WANDB_API_KEY="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"

NPROC_PER_NODE=2
WORLD_SIZE=1
RANK=0
MASTER_ADDR=0.0.0.0
MASTER_PORT=29502

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# model configs
MODEL_SIZE='2B'
MODEL_NAME="${MODEL_NAME:-models/Qwen3-VL-2B-Instruct}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/VLMEvalKit/outputs/swimbird_2b_eval/checkpoint-5774}"

DATA_PATH=(
    "SwimBird-SFT-92K-segment_0_plan/plan8/SwimBird-ZebraCoT/Zebra-CoT.json"
    "SwimBird-SFT-92K-segment_0_plan/plan8/SwimBird-ThinkMorph"
    "SwimBird-SFT-92K-segment_0_plan/plan8/SwimBird-MathCanvas"
    "SwimBird-SFT-92K-segment_0_plan/plan8/SwimBird-OpenMMReasoner"
)

RANDOM_SEED=42
GRAD_CHECK=True

GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NPROC_PER_NODE * WORLD_SIZE)))

# LLM-related params
LR=1e-5

# Latent-related params
LATENT_LOSS=mse
LATENT_LAMBDA=0.2
MAX_LATENT_TOKEN=32

MAX_TOKEN=16384
MIN_TOKEN=2

RUN_NAME="${RUN_NAME:-swimbird_singlenode_2b_segment_0_plan}"
OUTPUT_DIR="${OUTPUT_DIR:-swimbird_singlenode_2b_segment_0_plan}"

export PYTHONPATH=$(pwd)
echo "[train] MODEL_NAME=${MODEL_NAME}"
echo "[train] INIT_CHECKPOINT=${INIT_CHECKPOINT}"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] DATA_PATH=${DATA_PATH[*]}"
torchrun $DISTRIBUTED_ARGS \
    src/train/train.py \
    --run_name "$RUN_NAME" \
    --deepspeed scripts/zero2.json \
    --latent_loss $LATENT_LOSS \
    --checkpoint_name "$INIT_CHECKPOINT" \
    --model_id $MODEL_NAME \
    --data_path "${DATA_PATH[@]}" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate $LR \
    --latent_lambda $LATENT_LAMBDA \
    --max_latent_token $MAX_LATENT_TOKEN \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 32 * 32)) \
    --image_max_pixels $((MAX_TOKEN * 32 * 32)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --tf32 False \
    --gradient_checkpointing $GRAD_CHECK \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 30 \
    --dataloader_num_workers 8 \
    --random_seed $RANDOM_SEED \
    --report_to wandb
