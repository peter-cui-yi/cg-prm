#!/bin/bash
# CG-PRM Real Mini-Experiment
# Trains on real CLEVR + DocVQA traces and runs real model evaluation.
#
# Usage:
#   bash scripts/real_mini_exp.sh [GPU_SPEC]
#
# GPU_SPEC: comma-separated CUDA device indices (default: 0)
#   Single GPU:  bash scripts/real_mini_exp.sh 0
#   Multi-GPU:   bash scripts/real_mini_exp.sh 0,1,2,3
#
# Prerequisites (data already present):
#   data/mini_clevr/train_pairs.jsonl   (200 pairs)
#   data/mini_clevr/test_pairs.jsonl    ( 50 pairs)
#   data/mini_docvqa/train_pairs.jsonl  (120 pairs)
#   data/mini_docvqa/test_pairs.jsonl   ( 30 pairs)
#
# Outputs:
#   outputs/real_cg_prm/       LoRA adapter (pairwise CG-PRM)
#   outputs/real_pointwise/    LoRA adapter (pointwise baseline)
#   results/real_results.json  AUROC comparison + GO/MARGINAL/NO-GO decision
#   logs/real_cg_prm.log
#   logs/real_pointwise.log
#   logs/real_eval.log

set -euo pipefail

# ── GPU setup ────────────────────────────────────────────────────────────────
# Default: use all 4 GPUs for faster training
# Override: bash scripts/real_mini_exp.sh 0  (single GPU)
#           bash scripts/real_mini_exp.sh 0,1  (2 GPUs)
GPU_SPEC="${MINI_EXP_GPUS:-${1:-0,1,2,3}}"
IFS=',' read -ra _GPU_ARR <<< "$GPU_SPEC"
NUM_GPUS="${#_GPU_ARR[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_SPEC"

# DDP workaround for LoRA + gradient checkpointing
export TORCH_DDP_FIND_UNUSED_PARAMS_FALSE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Conda activation ─────────────────────────────────────────────────────────
if [[ -f "$HOME/anaconda3/bin/activate" ]]; then
  source "$HOME/anaconda3/bin/activate" nips27 2>/dev/null || \
    { source "$HOME/anaconda3/bin/activate" && conda activate nips27 2>/dev/null; } || true
fi

mkdir -p data/real_mini outputs logs results

echo "=============================================="
echo "CG-PRM Real Mini-Experiment"
echo "=============================================="
echo "Repo:               $REPO_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (NUM_GPUS=$NUM_GPUS)"
echo "Start:              $(date)"
echo ""

# ── Step 1: Merge real data ──────────────────────────────────────────────────
echo "=== Step 1: Merging real CLEVR + DocVQA data ==="

for split in train test; do
  clevr="data/mini_clevr/${split}_pairs.jsonl"
  docvqa="data/mini_docvqa/${split}_pairs.jsonl"
  out="data/real_mini/${split}_pairs.jsonl"
  if [[ ! -f "$clevr" || ! -f "$docvqa" ]]; then
    echo "ERROR: missing source file ($clevr or $docvqa)."
    echo "Run the real data generators first:"
    echo "  CLEVR_DIR=~/datasets/CLEVR_v1.0  python scripts/generate_mini_data_clevr.py"
    echo "  DOCVQA_DIR=~/datasets/DocVQA     python scripts/generate_mini_data_docvqa.py"
    exit 1
  fi
  cat "$clevr" "$docvqa" > "$out"
  n=$(wc -l < "$out")
  echo "  ${split}_pairs.jsonl: $n rows (CLEVR + DocVQA)"
done
echo ""

# ── Step 2: Train CG-PRM (pairwise) ─────────────────────────────────────────
echo "=== Step 2: Training CG-PRM (pairwise first-divergence) ==="
echo "Start: $(date)"
torchrun --nproc_per_node="$NUM_GPUS" --master_port=29700 \
  scripts/train_lora.py --config configs/real_cg_prm.json \
  2>&1 | tee logs/real_cg_prm.log
echo "End: $(date)"
echo ""

# ── Step 3: Train Pointwise baseline ────────────────────────────────────────
echo "=== Step 3: Training Pointwise baseline ==="
echo "Start: $(date)"
torchrun --nproc_per_node="$NUM_GPUS" --master_port=29701 \
  scripts/train_lora.py --config configs/real_pointwise.json \
  2>&1 | tee logs/real_pointwise.log
echo "End: $(date)"
echo ""

# ── Step 4: Evaluate ────────────────────────────────────────────────────────
echo "=== Step 4: Real model evaluation ==="
python scripts/evaluate_mini.py \
  --cg_prm    outputs/real_cg_prm \
  --pointwise outputs/real_pointwise \
  --test_data data/real_mini/test_pairs.jsonl \
  --output    results/real_results.json \
  2>&1 | tee logs/real_eval.log
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=============================================="
cat results/real_results.json
echo ""
echo "=============================================="
DECISION="$(python -c "import json; print(json.load(open('results/real_results.json'))['decision'])" 2>/dev/null || echo "unknown")"
echo "DECISION: $DECISION"
echo "Finished: $(date)"
echo "=============================================="
