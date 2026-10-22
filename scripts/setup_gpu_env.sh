#!/usr/bin/env bash
# Create the project's dedicated GPU environment without modifying other environments.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ENV_DIR="${ENV_DIR:-$ROOT/.venv-gpu}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-30}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Existing environment must use Python 3.11"'
"$PY" -m pip install --upgrade pip setuptools wheel
# Fetch CUDA wheels from PyTorch; resolve the remaining dependencies with PIP_INDEX_URL.
"$PY" -m pip install --no-deps torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r requirements/gpu-cu128.txt
"$PY" -m pip install -e '.[dev,vision,lora,api]' httpx tensorboard
"$PY" -m pip check
echo "Environment ready: $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
