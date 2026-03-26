#!/usr/bin/env bash

set -euo pipefail

DATASET_REPO="Accio-Lab/SwimBird-SFT-92K"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/SwimBird-SFT-92K"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/download_swimbird_data.sh [output_dir]

Arguments:
  output_dir   Optional. Where to download the Hugging Face dataset.
               Default: ${DEFAULT_OUTPUT_DIR}

Environment variables:
  HF_TOKEN     Optional Hugging Face token. Needed only if your environment
               cannot access the dataset anonymously.

What this script does:
  1. Downloads ${DATASET_REPO} from Hugging Face.
  2. Verifies the 4 expected dataset subdirectories exist.
  3. Rewrites JSON image paths to absolute paths with data_process.py.

Examples:
  bash scripts/download_swimbird_data.sh
  bash scripts/download_swimbird_data.sh /scratch/\$USER/SwimBird-SFT-92K
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

OUTPUT_DIR="${1:-${DEFAULT_OUTPUT_DIR}}"
OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_DIR}")"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

pick_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    echo "hf"
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli"
    return
  fi
  echo ""
}

require_cmd python

HF_CLI="$(pick_hf_cli)"
if [[ -z "${HF_CLI}" ]]; then
  echo "Hugging Face CLI not found." >&2
  echo "Install one of the following before running this script:" >&2
  echo "  pip install -U \"huggingface_hub[cli]\"" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Downloading dataset ${DATASET_REPO}"
echo "Output directory: ${OUTPUT_DIR}"

HF_ARGS=(
  download
  "${DATASET_REPO}"
  --repo-type dataset
  --local-dir "${OUTPUT_DIR}"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ARGS+=(--token "${HF_TOKEN}")
fi

"${HF_CLI}" "${HF_ARGS[@]}"

EXPECTED_SUBDIRS=(
  "SwimBird-ZebraCoT"
  "SwimBird-ThinkMorph"
  "SwimBird-MathCanvas"
  "SwimBird-OpenMMReasoner"
)

for subdir in "${EXPECTED_SUBDIRS[@]}"; do
  if [[ ! -d "${OUTPUT_DIR}/${subdir}" ]]; then
    echo "Expected directory not found after download: ${OUTPUT_DIR}/${subdir}" >&2
    echo "Inspect the downloaded dataset layout before training." >&2
    exit 1
  fi
done

for subdir in "${EXPECTED_SUBDIRS[@]}"; do
  echo "Rewriting image paths in ${OUTPUT_DIR}/${subdir}"
  python "${REPO_ROOT}/data_process.py" "${OUTPUT_DIR}/${subdir}"
done

cat <<EOF
Dataset is ready.

Training data root:
  ${OUTPUT_DIR}

Default train.sh expects:
  SwimBird-SFT-92K/SwimBird-ZebraCoT
  SwimBird-SFT-92K/SwimBird-ThinkMorph
  SwimBird-SFT-92K/SwimBird-MathCanvas
  SwimBird-SFT-92K/SwimBird-OpenMMReasoner

If you downloaded elsewhere, update DATA_PATH in scripts/train.sh accordingly.
EOF
