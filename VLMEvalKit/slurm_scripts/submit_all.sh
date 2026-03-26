#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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
    DynaMath
    WeMath
    MathVerse_MINI
    HRBench4K
    HRBench8K
    VStarBench
    MMStar
    RealWorldQA
)

if (( $# < 1 )); then
    echo "Usage: bash submit_all.sh <experiment_name> [benchmark1 benchmark2 ...]" >&2
    echo "Example: bash submit_all.sh pretrain DynaMath MMStar" >&2
    exit 1
fi

EXP_NAME="$1"
shift
LOG_ROOT="/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/VLMEvalKit/slurm_logs/${EXP_NAME}"
mkdir -p "${LOG_ROOT}"

if (( $# == 0 )); then
    TARGETS=("${ALL_BENCHMARKS[@]}")
else
    TARGETS=("$@")
fi

for benchmark in "${TARGETS[@]}"; do
    if [[ -z "${SCRIPT_MAP[$benchmark]+x}" ]]; then
        echo "Unknown benchmark: ${benchmark}" >&2
        echo "Available benchmarks: ${ALL_BENCHMARKS[*]}" >&2
        exit 1
    fi

    script="${SCRIPT_MAP[$benchmark]}"
    job_name="${EXP_NAME}_${benchmark}"
    log_stamp="$(date +%Y%m%d_%H%M%S)"
    out_log="${LOG_ROOT}/${benchmark}_${log_stamp}.out"
    echo "Submitting ${benchmark} with job name ${job_name}"
    sbatch \
        --job-name="${job_name}" \
        --output="${out_log}" \
        --error="${out_log}" \
        "${script}"
done
