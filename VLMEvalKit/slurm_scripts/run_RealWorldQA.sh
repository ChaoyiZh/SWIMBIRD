#!/bin/bash
#SBATCH --job-name=SwimBird_RealWorldQA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --partition=mri2020
#SBATCH --mail-user=chaoyiz@clemson.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/data/chaoyiz/workspace/code/SwimBird/VLMEvalKit/slurm_logs/%x.out
#SBATCH --error=/data/chaoyiz/workspace/code/SwimBird/VLMEvalKit/slurm_logs/%x.err

set -euo pipefail


PROJECT_ROOT="/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/VLMEvalKit"
cd "${PROJECT_ROOT}"

export LMUData=/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/datasets/VLMEval
mkdir -p "${LMUData}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"

CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=29507 \
    --nproc_per_node=1 \
    run.py \
    --data RealWorldQA \
    --model SwimBird-SFT-8B \
    --judge your_judge_model \
    --api-nproc 10 \
    --verbose
