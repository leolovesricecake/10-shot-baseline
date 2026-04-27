#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASETS_ROOT="${1:-${REPO_ROOT}/datasets}"
OUTPUT_ROOT="${2:-${REPO_ROOT}/results}"
MAX_SAMPLES="${3:-2}"
CUDA_DEVICE="${4:-${CUDA_VISIBLE_DEVICES:-0}}"
RETRIEVAL_MODEL_NAME="${5:-${BASELINE_RETRIEVAL_MODEL_NAME:-/mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf}}"

if [[ "${CUDA_DEVICE}" == *","* ]] || [[ "${CUDA_DEVICE}" =~ [[:space:]] ]]; then
  echo "ERROR: only a single CUDA device is allowed, got '${CUDA_DEVICE}'" >&2
  exit 1
fi
if [[ ! "${CUDA_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CUDA device must be a single integer index, got '${CUDA_DEVICE}'" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

MODELS=(glm4_9b llama3_1_8b)
BASELINES=(cot sc-cot sv-cot few-shot-cot random fixed)
FAILED=0

run_task() {
  local title="$1"
  shift
  echo "[RUN] ${title}"
  if "$@"; then
    echo "[OK] ${title}"
  else
    local code=$?
    echo "[FAIL] ${title} (exit=${code})" >&2
    FAILED=1
  fi
}

for model_alias in "${MODELS[@]}"; do
  for baseline_name in "${BASELINES[@]}"; do
    run_task "smoke model=${model_alias} baseline=${baseline_name}" \
      python "${REPO_ROOT}/run_baselines.py" \
        --baselines "${baseline_name}" \
        --datasets-root "${DATASETS_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --cuda-device "${CUDA_DEVICE}" \
        --retrieval-backend semantic \
        --retrieval-model-name "${RETRIEVAL_MODEL_NAME}" \
        --models "${model_alias}" \
        --max-samples "${MAX_SAMPLES}"
  done
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "[DONE] smoke completed with failures." >&2
  exit 1
fi
echo "[DONE] smoke all tasks succeeded."
