#!/bin/bash
# Complete Validation Experiment Pipeline
# Trains both models and evaluates them
# Usage: bash scripts/run_validation_complete.sh

set -e

cd /hpc2hdd/home/ycui785/cg-prm

echo "=============================================="
echo "CG-PRM Complete Validation Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Activate environment
source ~/anaconda3/bin/activate nips27
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# ===== Step 1: Train CG-PRM =====
echo "=== Step 1: Training CG-PRM (Pairwise) ==="
echo "Start: $(date)"
bash scripts/training/train_cg_prm.sh \
  configs/training/validation_cg_prm.json \
  4 \
  outputs/validation_cg_prm
echo "End: $(date)"
echo ""

# ===== Step 2: Train Pointwise Baseline =====
echo "=== Step 2: Training Pointwise Baseline ==="
echo "Start: $(date)"
bash scripts/training/train_cg_prm.sh \
  configs/training/validation_pointwise.json \
  4 \
  outputs/validation_pointwise
echo "End: $(date)"
echo ""

# ===== Step 3: Evaluate Both Models =====
echo "=== Step 3: Evaluating Models ==="
python scripts/evaluation/evaluate_mini.py \
  --cg_prm outputs/validation_cg_prm \
  --pointwise outputs/validation_pointwise \
  --test_data data/validation_5k/training_pairs/pairwise_val.jsonl \
  --output results/validation_results.json

echo ""
echo "=============================================="
echo "VALIDATION COMPLETE!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results:"
cat results/validation_results.json | python -m json.tool
echo ""
echo "Outputs:"
echo "  - Main results: results/validation_results.json"
echo "  - CG-PRM checkpoint: outputs/validation_cg_prm/"
echo "  - Pointwise checkpoint: outputs/validation_pointwise/"
echo ""
