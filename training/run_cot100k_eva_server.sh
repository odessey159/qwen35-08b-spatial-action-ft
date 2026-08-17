#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/qwen35-08b-spatial-action-ft"
PYTHON="${PROJECT_DIR}/.venv-train/bin/python"
RUN_DIR="${PROJECT_DIR}/outputs/qwen35-08b-cot-100k/v0-20260816-102046"
GEN_DIR="${RUN_DIR}/gpu-generation"
STATUS="${PROJECT_DIR}/run_logs/cot100k_eva.status"
EXPECTED_CF_PAIRS=1860
EXPECTED_PREDICTIONS=$((EXPECTED_CF_PAIRS * 2))

cd "${PROJECT_DIR}"
mkdir -p "${GEN_DIR}" run_logs
echo "running" >"${STATUS}"
finish() {
  local code=$?
  echo "${code}" >"${STATUS}"
}
trap finish EXIT

generate_cf() {
  local checkpoint=$1
  local output=$2
  local batch_size=$3
  local temporary="${output}.tmp"
  if [[ -f "${output}" ]] && [[ "$(wc -l <"${output}")" -eq "${EXPECTED_PREDICTIONS}" ]]; then
    echo "Skipping complete output: ${output}"
    return
  fi
  "${PYTHON}" -m training.generate_cpu_predictions \
    --checkpoint "${checkpoint}" \
    --val-file training/prepared/cot-100k/val.jsonl \
    --raw-data exp0/new100k_clean_data/samples.jsonl \
    --manifest training/prepared/cot-100k/manifest.json \
    --output "${temporary}" \
    --project-root "${PROJECT_DIR}" \
    --max-pairs "${EXPECTED_CF_PAIRS}" \
    --max-new-tokens 256 \
    --batch-size "${batch_size}" \
    --device cuda \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --overwrite
  [[ "$(wc -l <"${temporary}")" -eq "${EXPECTED_PREDICTIONS}" ]]
  mv -f "${temporary}" "${output}"
}

generate_cf \
  "${RUN_DIR}/checkpoint-11193" \
  "${GEN_DIR}/cf-all-step11193.jsonl" \
  16

generate_cf \
  "/model/ModelScope/Qwen/Qwen3.5-0.8B" \
  "${GEN_DIR}/cf-all-step0.jsonl" \
  16

"${PYTHON}" -m training.audit_eva \
  --raw-data exp0/new100k_clean_data/samples.jsonl \
  --manifest training/prepared/cot-100k/manifest.json \
  --output-dir "${RUN_DIR}/eva-audit-complete-cf" \
  --base-predictions "${GEN_DIR}/cf-all-step0.jsonl" \
  --base-scope current_100k_validation_cf_pairs \
  --base-cot-contract \
  --full-cot-predictions "${GEN_DIR}/cf-all-step11193.jsonl" \
  --current-scope current_100k_validation_cf_pairs
