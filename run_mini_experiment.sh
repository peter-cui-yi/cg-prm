#!/bin/bash
# CG-PRM Mini-Experiment - Full Pipeline
# Usage: ./run_mini_experiment.sh [GPU_ID]
# GPU_ID defaults to 0 if not specified

set -e

GPU_ID=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "CG-PRM Mini-Experiment"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo ""

# Create output directories
mkdir -p outputs logs results

# Step 1: Generate mini data
echo "=== Step 1: Generating mini dataset ==="
python scripts/data_generation/generate_mini_data.py
echo ""

# Step 2: Train CG-PRM (pairwise first-divergence)
echo "=== Step 2: Training CG-PRM (pairwise first-divergence) ==="
echo "Start: $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 scripts/training/train_lora.py \
    --config configs/training/mini_cg_prm.json 2>&1 | tee logs/cg_prm_train.log
echo "End: $(date)"
echo ""

# Step 3: Train Pointwise baseline
echo "=== Step 3: Training Pointwise baseline ==="
echo "Start: $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 scripts/training/train_lora.py \
    --config configs/training/mini_pointwise.json 2>&1 | tee logs/pointwise_train.log
echo "End: $(date)"
echo ""

# Step 4: Evaluate both models
echo "=== Step 4: Evaluating models ==="
python scripts/evaluation/evaluate_mini.py \
    --cg_prm outputs/mini_cg_prm \
    --pointwise outputs/mini_pointwise \
    --test_data data/mini/test_pairs.jsonl \
    --output results/mini_results.json
echo ""

# Step 5: Show results
echo "=============================================="
echo "RESULTS"
echo "=============================================="
cat results/mini_results.json
echo ""

# Extract decision
DECISION=$(python -c "import json; print(json.load(open('results/mini_results.json'))['decision'])")
echo "=============================================="
echo "DECISION: $DECISION"
echo "=============================================="

if [ "$DECISION" == "GO" ]; then
    echo "Proceed to full-scale experiment!"
    exit 0
elif [ "$DECISION" == "MARGINAL" ]; then
    echo "Weak signal - consider debugging or increasing data"
    exit 1
else
    echo "Hypothesis not supported - consider pivot"
    exit 2
fi
