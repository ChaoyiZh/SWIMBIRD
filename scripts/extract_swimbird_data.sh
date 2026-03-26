#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DATA_ROOT="${REPO_ROOT}/SwimBird-SFT-92K"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/extract_swimbird_data.sh [data_root]

Arguments:
  data_root   Optional. Root directory of SwimBird-SFT-92K.
              Default: ${DEFAULT_DATA_ROOT}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

DATA_ROOT="${1:-${DEFAULT_DATA_ROOT}}"
DATA_ROOT="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${DATA_ROOT}")"

if ! command -v unzip >/dev/null 2>&1; then
  echo "Missing required command: unzip" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

mapfile -t ZIP_FILES < <(find "${DATA_ROOT}" -type f -name '*.zip' | sort)

if [[ "${#ZIP_FILES[@]}" -eq 0 ]]; then
  echo "No zip files found under ${DATA_ROOT}"
  exit 0
fi

for zip_file in "${ZIP_FILES[@]}"; do
  target_dir="$(dirname "${zip_file}")"
  echo "Extracting ${zip_file} -> ${target_dir}"
  unzip -o -q "${zip_file}" -d "${target_dir}"
done

echo "Extraction complete."
