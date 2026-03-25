#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /ABS/PATH/TO/pipeline.json" >&2
  exit 1
fi

CONFIG_PATH="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[cg-prm] missing config: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

echo "[cg-prm] running pipeline with config: $CONFIG_PATH"
python "$ROOT_DIR/scripts/run_pipeline.py" --config "$CONFIG_PATH"

python - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
summary_path = config.get("summary_output")
if summary_path:
    summary_path = Path(summary_path)
    if not summary_path.is_absolute():
        summary_path = config_path.parent / summary_path
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("\n[cg-prm] benchmark summary")
        for benchmark, payload in summary.get("benchmarks", {}).items():
            status = payload.get("status", "unknown")
            print(f"  - {benchmark}: {status}")
            if status == "waiting_for_teacher_outputs":
                missing = payload.get("missing_teacher_outputs")
                print(f"    teacher outputs expected at: {missing}")
        training = summary.get("training_dataset", {})
        if training:
            print(f"  - training_dataset: {training.get('status', 'unknown')}")
PY
