#!/bin/bash

set -euo pipefail

MASTER_ADDR="$1"
shift

source /project/siyuh/common/chaoyi/miniconda3/etc/profile.d/conda.sh
conda activate swimbird

PROJECT_ROOT="${PROJECT_ROOT:-/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD}"
cd "${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/slurm_train_logs"

export PYTHONPATH="${PROJECT_ROOT}"
export WANDB_DISABLED=false
export WANDB_PROJECT="${WANDB_PROJECT:-SwimBird}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME:-swimbird}}"
export WANDB_WATCH="${WANDB_WATCH:-false}"
export WANDB_API_KEY="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"


# Enable additional NCCL diagnostics for multi-node runs.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NNODES="${SLURM_JOB_NUM_NODES:-4}"
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_PORT="${MASTER_PORT:-29502}"

MODEL_NAME="${MODEL_NAME:-models/Qwen3-VL-8B-Instruct}"
RUN_NAME="${RUN_NAME:-swimbird}"
OUTPUT_DIR="${OUTPUT_DIR:-swimbird}"
RANDOM_SEED="${RANDOM_SEED:-42}"
GRAD_CHECK="${GRAD_CHECK:-True}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * GPUS_PER_NODE * NNODES)))}"
LR="${LR:-1e-5}"
LATENT_LOSS="${LATENT_LOSS:-mse}"
LATENT_LAMBDA="${LATENT_LAMBDA:-0.2}"
MAX_LATENT_TOKEN="${MAX_LATENT_TOKEN:-32}"
MAX_TOKEN="${MAX_TOKEN:-16384}"
MIN_TOKEN="${MIN_TOKEN:-2}"

DATA_PATH=(
    "SwimBird-SFT-92K/SwimBird-ZebraCoT"
    "SwimBird-SFT-92K/SwimBird-ThinkMorph"
    "SwimBird-SFT-92K/SwimBird-MathCanvas"
    "SwimBird-SFT-92K/SwimBird-OpenMMReasoner"
)

echo "=========================================="
echo "Host: $(hostname)"
echo "Master addr: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo "Node rank: ${NODE_RANK}"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "=========================================="

set -x
torchrun \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    src/train/train.py \
    --run_name "${RUN_NAME}" \
    --deepspeed scripts/zero2.json \
    --latent_loss "${LATENT_LOSS}" \
    --model_id "${MODEL_NAME}" \
    --data_path "${DATA_PATH[@]}" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate "${LR}" \
    --latent_lambda "${LATENT_LAMBDA}" \
    --max_latent_token "${MAX_LATENT_TOKEN}" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${BATCH_PER_DEVICE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --image_min_pixels "$((MIN_TOKEN * 32 * 32))" \
    --image_max_pixels "$((MAX_TOKEN * 32 * 32))" \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --tf32 False \
    --gradient_checkpointing "${GRAD_CHECK}" \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --random_seed "${RANDOM_SEED}" \
    --report_to wandb \
    "$@"
