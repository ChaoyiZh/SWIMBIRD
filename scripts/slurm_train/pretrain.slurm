#!/bin/bash

#SBATCH --job-name=SwimBird_Pretrain
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --gpus-per-node=a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH -C interconnect_100g
#SBATCH --time=48:00:00
#SBATCH --partition=mri2020
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chaoyiz@clemson.edu
#SBATCH --output=/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/slurm_logs/%x_%j.out
#SBATCH --error=/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/slurm_logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

mkdir -p /project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/slurm_logs

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Master node: ${master_addr}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: 2"
echo "=========================================="

srun bash "${SCRIPT_DIR}/pretrain_slurm.sh" "${master_addr}" "$@"
