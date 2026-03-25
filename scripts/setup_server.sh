#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
INSTALL_BITSANDBYTES="${INSTALL_BITSANDBYTES:-0}"

echo "[cg-prm] root: $ROOT_DIR"
echo "[cg-prm] venv: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" == "1" ]]; then
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    python -m pip install --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
  else
    python -m pip install "$TORCH_SPEC"
  fi
fi

python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "$INSTALL_BITSANDBYTES" == "1" ]]; then
  python -m pip install bitsandbytes
fi

echo
echo "[cg-prm] environment ready"
echo "[cg-prm] activate with: source \"$VENV_DIR/bin/activate\""
echo "[cg-prm] first pipeline run: bash \"$ROOT_DIR/scripts/first_run.sh\" /ABS/PATH/TO/pipeline.json"
