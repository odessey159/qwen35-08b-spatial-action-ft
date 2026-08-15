#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/qwen35-08b-spatial-action-ft"
PYTHON312="/usr/local/miniconda3/envs/py312/bin/python"
VENV_DIR="${PROJECT_DIR}/.venv-train"
export CUDA_HOME="/usr/local/cuda-13.2"
export TORCH_CUDA_ARCH_LIST="8.9"
export FLASH_ATTN_CUDA_ARCHS="80"
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}"
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
if ! "${VENV_DIR}/bin/python" -c "import causal_conv1d"; then
  CAUSAL_CONV1D_DIR="$(mktemp -d)"
  git clone --filter=blob:none \
    https://github.com/Dao-AILab/causal-conv1d "${CAUSAL_CONV1D_DIR}"
  git -C "${CAUSAL_CONV1D_DIR}" checkout --detach \
    3a4c88e599cd7dec333cac727bd59f2a41a8aad5
  # This pinned setup.py hard-codes every CUDA 13 architecture and ignores
  # TORCH_CUDA_ARCH_LIST. Compile only the RTX 4090 target used by this server.
  sed -i '179,199c\
        cc_flag.extend(["-gencode", "arch=compute_89,code=sm_89"])' \
    "${CAUSAL_CONV1D_DIR}/setup.py"
  "${VENV_DIR}/bin/python" -m pip install --upgrade \
    "${CAUSAL_CONV1D_DIR}" --no-build-isolation
  rm -rf "${CAUSAL_CONV1D_DIR}"
fi
if ! "${VENV_DIR}/bin/python" -c "import flash_attn"; then
  "${VENV_DIR}/bin/python" -m pip install \
    "flash-attn==2.8.3" --no-build-isolation
fi

"${VENV_DIR}/bin/python" -m training.cli \
  --config "${PROJECT_DIR}/training/config.server.json" check-runtime
