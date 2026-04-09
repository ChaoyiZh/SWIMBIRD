#!/bin/bash

#SBATCH --job-name=SwimBird_Pretrain_SingleNode_2B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=mri2020
#SBATCH --mail-user=chaoyiz@clemson.edu
#SBATCH --output=/project/siyuh/common/chaoyi/workspace/code/SwimBird/slurm_pretrain_singlenode_2b_logs/%x_%j.out
#SBATCH --error=/project/siyuh/common/chaoyi/workspace/code/SwimBird/slurm_pretrain_singlenode_2b_logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/siyuh/common/chaoyi/workspace/code/SwimBird}"
LOG_DIR="${PROJECT_ROOT}/slurm_pretrain_singlenode_2b_logs"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/slurm_train_2b"
LAUNCH_SCRIPT="${SCRIPT_DIR}/pretrain_slurm.sh"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Node list: ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "GPUs requested: 2 x A100"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-16}"
echo "Memory: 128GB"
echo "=========================================="

srun bash "${LAUNCH_SCRIPT}" "$@"
