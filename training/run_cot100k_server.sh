#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/qwen35-08b-spatial-action-ft"
PYTHON="${PROJECT_DIR}/.venv-train/bin/python"
CONFIG="${PROJECT_DIR}/training/config.cot.100k.server.json"
STATUS="${PROJECT_DIR}/run_logs/cot100k_train.status"

cd "${PROJECT_DIR}"
mkdir -p run_logs
if [[ -e "${STATUS}" ]]; then
  echo "Refusing to reuse existing status file: ${STATUS}" >&2
  exit 2
fi
if [[ -d outputs/qwen35-08b-cot-100k ]] &&
   find outputs/qwen35-08b-cot-100k -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse non-empty output directory" >&2
  exit 3
fi

echo "running" >"${STATUS}"
finish() {
  local code=$?
  echo "${code}" >"${STATUS}"
}
trap finish EXIT

"${PYTHON}" -m training.cli --config "${CONFIG}" validate
"${PYTHON}" -m training.cli --config "${CONFIG}" check-runtime
"${PYTHON}" -m training.cli --config "${CONFIG}" show-command
"${PYTHON}" -m training.cli --config "${CONFIG}" train
