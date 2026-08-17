#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/qwen35-08b-spatial-action-ft"
PYTHON="${PROJECT_DIR}/.venv-train/bin/python"
RUN_DIR="${PROJECT_DIR}/outputs/qwen35-08b-cot-100k/v0-20260816-102046"
OUTPUT_DIR="${RUN_DIR}/section-loss-eval-gpu-bf16-step0-full"
STATUS="${PROJECT_DIR}/run_logs/cot100k_step0_full_section.status"

cd "${PROJECT_DIR}"
mkdir -p run_logs
if [[ -e "${STATUS}" ]]; then
  echo "Refusing to reuse existing status file: ${STATUS}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
  exit 3
fi

echo "running" >"${STATUS}"
finish() {
  local code=$?
  echo "${code}" >"${STATUS}"
}
trap finish EXIT

"${PYTHON}" -m training.evaluate_section_losses \
  --run-dir "${RUN_DIR}" \
  --base-model-dir /model/ModelScope/Qwen/Qwen3.5-0.8B \
  --val-file training/prepared/cot-100k/val.jsonl \
  --checkpoint-steps 0 \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_DIR}" \
  --max-samples 0 \
  --sample-seed 42 \
  --device cuda \
  --dtype bfloat16 \
  --cpu-threads 8
