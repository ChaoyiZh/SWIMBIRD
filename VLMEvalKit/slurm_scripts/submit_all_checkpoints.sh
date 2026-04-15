#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
CONFIG_FILE="${SCRIPT_DIR}/../vlmeval/config.py"

declare -A SCRIPT_MAP=(
    [DynaMath]=run_DynaMath.sh
    [WeMath]=run_WeMath.sh
    [MathVerse_MINI]=run_MathVerse_MINI.sh
    [HRBench4K]=run_HRBench4K.sh
    [HRBench8K]=run_HRBench8K.sh
    [VStarBench]=run_VStarBench.sh
    [MMStar]=run_MMStar.sh
    [RealWorldQA]=run_RealWorldQA.sh
)

ALL_BENCHMARKS=(
    HRBench4K
    HRBench8K
    VStarBench
    RealWorldQA
)
# ALL_BENCHMARKS=(
#     DynaMath
#     WeMath
#     MathVerse_MINI
#     HRBench4K
#     HRBench8K
#     VStarBench
#     MMStar
#     RealWorldQA
# )


usage() {
    cat >&2 <<'EOF'
Usage: bash submit_all_checkpoints.sh <experiment_name> [checkpoint_root] [benchmark1 benchmark2 ...]

Examples:
  bash submit_all_checkpoints.sh swimbird_8b_eval
  bash submit_all_checkpoints.sh swimbird_8b_eval /project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird
  bash submit_all_checkpoints.sh swimbird_8b_eval /project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird DynaMath MMStar

Notes:
  - The script submits one slurm job per (checkpoint, benchmark).
  - The script derives model aliases from checkpoint names and checkpoint root.
  - Example: swimbird/checkpoint-400 -> SwimBird-SFT-8B_ckpt400
  - Example: swimbird_singlenode_2b/checkpoint-4200 -> SwimBird-SFT-2B_ckpt4200
  - Example: swimbird_singlenode_2b_thought0_latent/checkpoint-600 -> SwimBird-SFT-2B-Thought0-Latent_ckpt600
  - The checkpoint aliases must exist in vlmeval/config.py.
EOF
    exit 1
}

if (( $# < 1 )); then
    usage
fi

EXP_NAME="$1"
shift

DEFAULT_CHECKPOINT_ROOT="/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b"
CHECKPOINT_ROOT="${DEFAULT_CHECKPOINT_ROOT}"

if (( $# >= 1 )) && [[ -d "$1" ]]; then
    CHECKPOINT_ROOT="$1"
    shift
fi

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    echo "Checkpoint root does not exist: ${CHECKPOINT_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file does not exist: ${CONFIG_FILE}" >&2
    exit 1
fi

checkpoint_root_name="$(basename "${CHECKPOINT_ROOT}")"
if [[ "${checkpoint_root_name}" == "swimbird" ]]; then
    MODEL_PREFIX="SwimBird-SFT-8B_ckpt"
elif [[ "${checkpoint_root_name}" == "swimbird_singlenode_2b" ]]; then
    MODEL_PREFIX="SwimBird-SFT-2B_ckpt"
elif [[ "${checkpoint_root_name}" == "swimbird_singlenode_2b_thought0_latent" ]]; then
    MODEL_PREFIX="SwimBird-SFT-2B-Thought0-Latent_ckpt"
elif [[ "${checkpoint_root_name}" == "swimbird_singlenode_2b_best_ckpt_segment_0_plan" ]]; then
    MODEL_PREFIX="SwimBird-SFT-2B-Best-Ckpt-Segment-0-Plan_ckpt"
else
    echo "Unsupported checkpoint root: ${CHECKPOINT_ROOT}" >&2
    echo "Expected one of:" >&2
    echo "  /project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird" >&2
    echo "  /project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b" >&2
    echo "  /project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b_thought0_latent" >&2
    echo "  /project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b_best_ckpt_segment_0_plan" >&2
    exit 1
fi

LOG_ROOT="/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/VLMEvalKit/slurm_logs/${EXP_NAME}"
EVAL_ROOT_BASE="/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/VLMEvalKit/outputs/${EXP_NAME}"
mkdir -p "${LOG_ROOT}"
mkdir -p "${EVAL_ROOT_BASE}"

if (( $# == 0 )); then
    TARGETS=("${ALL_BENCHMARKS[@]}")
else
    TARGETS=("$@")
fi

mapfile -t CHECKPOINT_DIRS < <(find "${CHECKPOINT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)

if (( ${#CHECKPOINT_DIRS[@]} == 0 )); then
    echo "No checkpoint-* directories found under ${CHECKPOINT_ROOT}" >&2
    exit 1
fi

mapfile -t REGISTERED_STEPS < <(
    MODEL_PREFIX="${MODEL_PREFIX}" CONFIG_FILE="${CONFIG_FILE}" python - <<'PY'
import os
import pathlib
import re
import sys

model_prefix = os.environ["MODEL_PREFIX"]
config_path = pathlib.Path(os.environ["CONFIG_FILE"])
text = config_path.read_text()

prefix_to_pattern = {
    "SwimBird-SFT-8B_ckpt": r"for step in \(([^)]*)\):\n\s*swimbird_series\[f'SwimBird-SFT-8B_ckpt\{step\}'\]",
    "SwimBird-SFT-2B_ckpt": r"for step in \(([^)]*)\):\n\s*swimbird_series\[f'SwimBird-SFT-2B_ckpt\{step\}'\]",
    "SwimBird-SFT-2B-Thought0-Latent_ckpt": r"for step in \(([^)]*)\):\n\s*swimbird_series\[f'SwimBird-SFT-2B-Thought0-Latent_ckpt\{step\}'\]",
    "SwimBird-SFT-2B-Best-Ckpt-Segment-0-Plan_ckpt": r"for step in \(([^)]*)\):\n\s*swimbird_series\[f'SwimBird-SFT-2B-Best-Ckpt-Segment-0-Plan_ckpt\{step\}'\]",
}

pattern = prefix_to_pattern.get(model_prefix)
if pattern is None:
    sys.exit(f"Unsupported model prefix: {model_prefix}")

match = re.search(pattern, text)
if match is None:
    sys.exit(f"Could not find registered checkpoints for prefix: {model_prefix}")

for raw_step in match.group(1).split(","):
    step = raw_step.strip()
    if step:
        print(step)
PY
)

if (( ${#REGISTERED_STEPS[@]} == 0 )); then
    echo "No registered checkpoints found in ${CONFIG_FILE} for prefix ${MODEL_PREFIX}" >&2
    exit 1
fi

declare -A REGISTERED_STEP_SET=()
for step in "${REGISTERED_STEPS[@]}"; do
    REGISTERED_STEP_SET["$step"]=1
done

FILTERED_CHECKPOINT_DIRS=()
for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint_name="$(basename "${checkpoint_dir}")"
    step="${checkpoint_name#checkpoint-}"
    if [[ -n "${REGISTERED_STEP_SET[$step]+x}" ]]; then
        FILTERED_CHECKPOINT_DIRS+=("${checkpoint_dir}")
    else
        echo "Skipping unregistered checkpoint: ${checkpoint_name}" >&2
    fi
done

CHECKPOINT_DIRS=("${FILTERED_CHECKPOINT_DIRS[@]}")

if (( ${#CHECKPOINT_DIRS[@]} == 0 )); then
    echo "No registered checkpoint-* directories found under ${CHECKPOINT_ROOT}" >&2
    exit 1
fi

for benchmark in "${TARGETS[@]}"; do
    if [[ -z "${SCRIPT_MAP[$benchmark]+x}" ]]; then
        echo "Unknown benchmark: ${benchmark}" >&2
        echo "Available benchmarks: ${ALL_BENCHMARKS[*]}" >&2
        exit 1
    fi
done

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint_name="$(basename "${checkpoint_dir}")"
    step="${checkpoint_name#checkpoint-}"
    model_name="${MODEL_PREFIX}${step}"
    step_log_root="${LOG_ROOT}/${checkpoint_name}"
    step_eval_root="${EVAL_ROOT_BASE}/${checkpoint_name}"
    mkdir -p "${step_log_root}"
    mkdir -p "${step_eval_root}"

    for benchmark in "${TARGETS[@]}"; do
        script="${SCRIPT_MAP[$benchmark]}"
        job_name="${EXP_NAME}_${benchmark}_s${step}"
        log_stamp="$(date +%Y%m%d_%H%M%S)"
        out_log="${step_log_root}/${benchmark}_${log_stamp}.out"

        echo "Submitting ${benchmark} for ${checkpoint_name}"
        echo "  model name: ${model_name}"
        echo "  eval root : ${step_eval_root}"
        sbatch \
            --job-name="${job_name}" \
            --output="${out_log}" \
            --error="${out_log}" \
            --export="ALL,MMEVAL_ROOT=${step_eval_root}" \
            "${script}" "${model_name}"
    done
done
