#!/usr/bin/env bash
# Project-local, resumable installation. Does not start training.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$project_root"
export PATH="$project_root/.venv-train/bin:$PATH"
export UV_CACHE_DIR="$project_root/runtime/cache/uv"
export UV_PYTHON_INSTALL_DIR="$project_root/runtime/python"
export UV_PYTHON_BIN_DIR="$project_root/runtime/bin"
export UV_NO_CONFIG=1
export UV_SYSTEM_CERTS=true
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=5
export PIP_CACHE_DIR="$project_root/runtime/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export XDG_CACHE_HOME="$project_root/runtime/cache/xdg"
export XDG_CONFIG_HOME="$project_root/runtime/config"
export XDG_DATA_HOME="$project_root/runtime/share"
export XDG_STATE_HOME="$project_root/runtime/state"
export TORCH_HOME="$project_root/runtime/cache/torch"
export TORCH_EXTENSIONS_DIR="$project_root/runtime/cache/torch_extensions"
export CUDA_CACHE_PATH="$project_root/runtime/cache/cuda"
export HF_HOME="$project_root/runtime/cache/huggingface"
export MPLCONFIGDIR="$project_root/runtime/cache/matplotlib"
export NUMBA_CACHE_DIR="$project_root/runtime/cache/numba"
export WANDB_DIR="$project_root/runtime/cache/wandb"
export WANDB_CONFIG_DIR="$project_root/runtime/cache/wandb_config"
export WANDB_CACHE_DIR="$project_root/runtime/cache/wandb_cache"
export JUPYTER_CONFIG_DIR="$project_root/runtime/config/jupyter"
export JUPYTER_DATA_DIR="$project_root/runtime/share/jupyter"
export IPYTHONDIR="$project_root/runtime/config/ipython"
export TMPDIR="$project_root/runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-4}"

if [[ "${1:-}" == "--print-env" ]]; then
  for key in UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR UV_NO_CONFIG UV_SYSTEM_CERTS \
    UV_HTTP_TIMEOUT UV_HTTP_RETRIES PIP_CACHE_DIR PIP_CONFIG_FILE PIP_DISABLE_PIP_VERSION_CHECK \
    XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME TORCH_HOME TORCH_EXTENSIONS_DIR \
    CUDA_CACHE_PATH HF_HOME MPLCONFIGDIR NUMBA_CACHE_DIR WANDB_DIR WANDB_CONFIG_DIR WANDB_CACHE_DIR \
    JUPYTER_CONFIG_DIR JUPYTER_DATA_DIR IPYTHONDIR TMPDIR TMP TEMP PYTHONDONTWRITEBYTECODE \
    PYTHONNOUSERSITE MAX_JOBS PATH; do
    printf 'export %s=%q\n' "$key" "${!key}"
  done
  exit 0
fi
mode="${1:-}"
minimal=false
if [[ "${2:-}" == "--minimal" ]]; then minimal=true; fi
if [[ "$mode" != "--install" && "$mode" != "--check" ]]; then
  echo "Python 3.10 / Torch 2.4.0+cu124 / gsplat 1.4.0 / Nerfstudio 1.1.5"
  echo "Usage: bash scripts/bootstrap_training.sh --install [--minimal]"
  echo "       bash scripts/bootstrap_training.sh --check [--minimal]"
  echo '       eval "$(bash scripts/bootstrap_training.sh --print-env)"'
  exit 0
fi
if [[ $# -gt 2 || ( $# -eq 2 && "${2:-}" != "--minimal" ) ]]; then
  echo "Unknown argument" >&2
  exit 2
fi
mkdir -p -- "$TMPDIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
  "$UV_PYTHON_BIN_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$XDG_STATE_HOME"
python="$project_root/.venv-train/bin/python"
constraints="$project_root/configs/training.constraints.txt"
if [[ "$mode" == "--install" ]]; then
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 2; }
  # Retries reuse the same interpreter, environment and download cache.
  uv python install 3.10 --no-bin
  if [[ ! -e "$project_root/.venv-train" ]]; then
    uv venv --managed-python --python 3.10 "$project_root/.venv-train"
  fi
  "$python" -c 'import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3,10)'
  uv pip install --python "$python" --default-index https://pypi.org/simple \
    --constraint "$constraints" pip setuptools wheel ninja
  echo "Installing official Torch cu124 wheels (several GB on first install)."
  uv pip install --python "$python" --torch-backend cu124 --default-index https://pypi.org/simple \
    --constraint "$constraints" torch torchvision
  uv pip install --python "$python" --default-index https://pypi.org/simple \
    --constraint "$constraints" gsplat numpy
  if ! "$minimal"; then
    echo "Installing the complete Nerfstudio dependency set."
    uv pip install --python "$python" --torch-backend cu124 --default-index https://pypi.org/simple \
      --constraint "$constraints" nerfstudio
    uv pip check --python "$python"
  fi
  uv pip freeze --python "$python" > "$project_root/runtime/training.freeze.txt"
fi
probe_args=()
if "$minimal"; then probe_args+=(--minimal); fi
"$python" "$project_root/scripts/gpu_preflight.py" "${probe_args[@]}" \
  --output "$project_root/runtime/gpu_preflight.json"
if ! "$minimal"; then
  "$project_root/.venv-train/bin/ns-train" splatfacto --help \
    > "$project_root/runtime/ns-train-splatfacto-help.txt"
fi
echo "CUDA forward/backward passed; evidence: runtime/gpu_preflight.json"
