#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/qwen35-08b-spatial-action-ft"
PYTHON312="/usr/local/miniconda3/envs/py312/bin/python"
VENV_DIR="${PROJECT_DIR}/.venv-train"
export CUDA_HOME="/usr/local/cuda-13.2"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${PYTHON312}" ]]; then
  echo "Python 3.12 not found at ${PYTHON312}" >&2
  exit 1
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

"${PYTHON312}" -m venv --system-site-packages "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel ninja packaging
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements-training.txt"
"${VENV_DIR}/bin/python" -m pip install --upgrade \
  "flash-linear-attention>=0.4.2" --no-build-isolation
"${VENV_DIR}/bin/python" -m pip install --upgrade \
  "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d" --no-build-isolation
"${VENV_DIR}/bin/python" -m pip install \
  "flash-attn==2.8.3" --no-build-isolation

"${VENV_DIR}/bin/python" -m training.cli \
  --config "${PROJECT_DIR}/training/config.server.json" check-runtime
