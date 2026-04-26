#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASETS_ROOT="${1:-${REPO_ROOT}/datasets}"
OUTPUT_ROOT="${2:-${REPO_ROOT}/results}"
MAX_SAMPLES="${3:-2}"

python "${REPO_ROOT}/run_cot.py" \
  --datasets-root "${DATASETS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --models glm4_9b llama3_1_8b \
  --max-samples "${MAX_SAMPLES}" \
  --skip-missing-datasets
