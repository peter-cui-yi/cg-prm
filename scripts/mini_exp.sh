#!/bin/bash
# CG-PRM Mini-Experiment - Quick Start (same pipeline as run_mini_experiment.sh)
#
# Usage:
#   bash scripts/mini_exp.sh [GPU_SPEC]
#
# GPU_SPEC — comma-separated CUDA device indices (default: 0). Count sets DDP width.
#   Single GPU:  bash scripts/mini_exp.sh
#                bash scripts/mini_exp.sh 0
#   Multi-GPU:   bash scripts/mini_exp.sh 0,1
#                bash scripts/mini_exp.sh 0,1,2,3
#   Env override: MINI_EXP_GPUS=0,1,2 bash scripts/mini_exp.sh
#
# Effective batch scales with NUM_GPUS × per_device_batch × grad_accum (see configs).
#
# Data (Step 1):
#   MINI_DATA_MODE=mock (default)  — synthetic traces via generate_mini_data.py → data/mini/
#   MINI_DATA_MODE=real            — real CLEVR + DocVQA generators, then merge into data/mini/
#     Set CLEVR_DIR / DOCVQA_DIR if defaults are wrong (see generate_mini_data_*.py).

set -euo pipefail

GPU_SPEC="${MINI_EXP_GPUS:-${1:-0}}"
IFS=',' read -ra _GPU_ARR <<< "$GPU_SPEC"
NUM_GPUS="${#_GPU_ARR[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "ERROR: need at least one GPU index in GPU_SPEC (got: ${GPU_SPEC})" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_SPEC"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MINI_DATA_MODE="${MINI_DATA_MODE:-mock}"

# Activate conda env (works in both interactive and nohup/non-interactive shells)
if [[ -f "$HOME/anaconda3/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/anaconda3/bin/activate" nips27 2>/dev/null || \
    source "$HOME/anaconda3/bin/activate" && conda activate nips27 2>/dev/null || true
fi

# Clean up stuck training runs (narrow pattern)
echo "Cleaning up prior train jobs on this host..."
pkill -f "scripts/train_lora.py" 2>/dev/null || true
sleep 1

mkdir -p outputs logs results data/mini

echo "=============================================="
echo "CG-PRM Mini-Experiment"
echo "=============================================="
echo "Repo: $REPO_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (NUM_GPUS=$NUM_GPUS)"
echo "MINI_DATA_MODE: $MINI_DATA_MODE"
echo "Start: $(date)"
echo ""

# Step 1: Generate data
echo "=== Step 1: Mini dataset → data/mini/ ==="
if [[ "$MINI_DATA_MODE" == "real" ]]; then
  export CLEVR_DIR="${CLEVR_DIR:-$HOME/datasets/CLEVR_v1.0}"
  export DOCVQA_DIR="${DOCVQA_DIR:-$HOME/datasets/DocVQA}"
  echo "CLEVR_DIR=$CLEVR_DIR"
  echo "DOCVQA_DIR=$DOCVQA_DIR"
  python scripts/generate_mini_data_clevr.py
  python scripts/generate_mini_data_docvqa.py
  for f in data/mini_clevr/train_pairs.jsonl data/mini_docvqa/train_pairs.jsonl \
           data/mini_clevr/test_pairs.jsonl data/mini_docvqa/test_pairs.jsonl; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: expected file missing after generation: $f"
      exit 1
    fi
  done
  cat data/mini_clevr/train_pairs.jsonl data/mini_docvqa/train_pairs.jsonl > data/mini/train_pairs.jsonl
  cat data/mini_clevr/test_pairs.jsonl data/mini_docvqa/test_pairs.jsonl > data/mini/test_pairs.jsonl
  echo "Merged CLEVR + DocVQA pairs into data/mini/"
else
  python scripts/generate_mini_data.py
fi
echo ""

# Step 2: Train CG-PRM
echo "=== Step 2: Training CG-PRM (pairwise) ==="
echo "Start: $(date)"
torchrun --nproc_per_node="$NUM_GPUS" --master_port=29600 scripts/train_lora.py \
  --config configs/mini_cg_prm.json 2>&1 | tee logs/cg_prm_train.log
echo "End: $(date)"
echo ""

# Step 3: Train Pointwise
echo "=== Step 3: Training Pointwise ==="
echo "Start: $(date)"
torchrun --nproc_per_node="$NUM_GPUS" --master_port=29601 scripts/train_lora.py \
  --config configs/mini_pointwise.json 2>&1 | tee logs/pointwise_train.log
echo "End: $(date)"
echo ""

# Step 4: Evaluate
echo "=== Step 4: Evaluation ==="
python scripts/evaluate_mini.py \
  --cg_prm outputs/mini_cg_prm \
  --pointwise outputs/mini_pointwise \
  --test_data data/mini/test_pairs.jsonl \
  --output results/mini_results.json

echo ""
echo "=============================================="
cat results/mini_results.json
echo ""
echo "=============================================="

DECISION="$(python -c "import json; print(json.load(open('results/mini_results.json'))['decision'])" 2>/dev/null || echo "unknown")"
echo "DECISION: $DECISION"
echo "Finished: $(date)"
echo "=============================================="
