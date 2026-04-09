#!/bin/bash

set -euo pipefail

source /home/chaoyiz/miniconda3/etc/profile.d/conda.sh
conda activate swimbird

PROJECT_ROOT="${PROJECT_ROOT:-/project/siyuh/common/chaoyi/workspace/code/SwimBird}"
LOG_DIR="${PROJECT_ROOT}/slurm_pretrain_singlenode_2b_logs"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Host: $(hostname)"
echo "GPUs requested: ${SLURM_GPUS_ON_NODE:-2}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-8}"
echo "=========================================="

bash "${PROJECT_ROOT}/scripts/train_2b.sh" "$@"
