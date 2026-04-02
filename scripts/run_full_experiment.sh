#!/bin/bash
# Full-scale CG-PRM experiment orchestration with multi-GPU support
# Usage: bash scripts/run_full_experiment.sh [NUM_GPUS]
# Default: 4 GPUs

set -e

NUM_GPUS="${1:-4}"
TEACHER_GPUS=$NUM_GPUS
TRAINING_GPUS=$NUM_GPUS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "CG-PRM Full-Scale Experiment (Multi-GPU)"
echo "=============================================="
echo "Start time: $(date)"
echo "GPUs: $NUM_GPUS x A800 80GB"
echo "Working directory: $(pwd)"
echo ""

# Create directories
mkdir -p outputs logs results data/full

# Activate environment and set PYTHONPATH
source ~/anaconda3/bin/activate nips27
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# ===== Phase 1: Model Download =====
echo "=== Phase 1: Model Download ==="
if [ ! -f "/hpc2hdd/home/ycui785/model/qwen3vl-32b-thinking/config.json" ]; then
    echo "Downloading teacher model..."
    bash scripts/inference/download_teacher_model.sh
else
    echo "✓ Teacher model already exists"
fi
echo ""

# ===== Phase 2: Dataset Generation (uses all GPUs) =====
echo "=== Phase 2: Dataset Generation ==="

# Check if datasets exist
if [ ! -d "/hpc2hdd/home/ycui785/datasets/CLEVR/CLEVR_v1.0" ]; then
    echo "ERROR: CLEVR dataset not found"
    echo "Please download to /hpc2hdd/home/ycui785/datasets/CLEVR"
    exit 1
fi

if [ ! -d "/hpc2hdd/home/ycui785/datasets/DocVQA" ]; then
    echo "ERROR: DocVQA dataset not found"
    echo "Please download to /hpc2hdd/home/ycui785/datasets/DocVQA"
    exit 1
fi

# Check if data already generated
if [ -f "data/full/training_pairs/pairwise_train.jsonl" ] && [ -f "data/full/training_pairs/pointwise_train.jsonl" ]; then
    echo "✓ Full dataset already exists, skipping generation"
    ls -lh data/full/training_pairs/
else
    echo "Generating full-scale dataset with $NUM_GPUS GPUs..."
    NUM_GPUS=$NUM_GPUS bash scripts/data_generation/generate_full_data_parallel.sh
fi

echo ""

# ===== Phase 3: Training (uses all GPUs per model) =====
echo "=== Phase 3: Training ==="

# Train CG-PRM (pairwise) - use all 4 GPUs
if [ ! -d "outputs/full_cg_prm/checkpoint-1000" ]; then
    echo "Training CG-PRM (pairwise) on $TRAINING_GPUS GPUs..."
    echo "Start: $(date)"
    bash scripts/training/train_cg_prm.sh \
        configs/training/full_cg_prm.json \
        $TRAINING_GPUS \
        outputs/full_cg_prm
    echo "End: $(date)"
    echo ""
else
    echo "✓ CG-PRM model already trained"
fi

# Train Pointwise baseline - use all 4 GPUs
if [ ! -d "outputs/full_pointwise/checkpoint-1000" ]; then
    echo "Training Pointwise baseline on $TRAINING_GPUS GPUs..."
    echo "Start: $(date)"
    bash scripts/training/train_cg_prm.sh \
        configs/training/full_pointwise.json \
        $TRAINING_GPUS \
        outputs/full_pointwise
    echo "End: $(date)"
    echo ""
else
    echo "✓ Pointwise model already trained"
fi

# ===== Phase 4: Evaluation =====
echo "=== Phase 4: Evaluation ==="

echo "Running comprehensive evaluation..."
source ~/anaconda3/bin/activate nips27
python scripts/evaluation/evaluate_full.py \
    --cg_prm outputs/full_cg_prm \
    --pointwise outputs/full_pointwise \
    --test_pairs data/full/training_pairs/pairwise_val.jsonl \
    --output_dir results/full \
    --step_analysis \
    --corruption_ablation

echo ""

echo "Running corruption family ablation..."
python scripts/evaluation/ablation_by_corruption.py \
    --test_pairs data/full/training_pairs/pairwise_val.jsonl \
    --output results/full/corruption_ablation.json \
    --generate-latex

echo ""

echo "Aggregating results..."
python scripts/evaluation/aggregate_results.py \
    --input_dir results/full \
    --output results/full_experiment_summary.json \
    --generate-latex

echo ""

# ===== Summary =====
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results:"
cat results/full_experiment_summary.json | python -m json.tool
echo ""
echo "Outputs:"
echo "  - Main results: results/full/full_results.json"
echo "  - Summary: results/full_experiment_summary.json"
echo "  - Checkpoints: outputs/full_cg_prm/, outputs/full_pointwise/"
echo "  - Logs: logs/training_full_cg_prm.log, logs/training_full_pointwise.log"
echo ""
echo "GPU Utilization:"
echo "  - Teacher inference: $TEACHER_GPUS GPUs (vLLM tensor parallel)"
echo "  - Training: $TRAINING_GPUS GPUs per model (DDP)"
echo ""
