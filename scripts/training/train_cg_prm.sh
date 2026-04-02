#!/bin/bash
# Train CG-PRM model with multi-GPU support
# Usage: bash scripts/training/train_cg_prm.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR]

set -e

CONFIG="${1:-configs/training/full_cg_prm.json}"
NUM_GPUS="${2:-4}"
OUTPUT_DIR="${3:-outputs/full_cg_prm}"

echo "=============================================="
echo "CG-PRM Multi-GPU Training"
echo "=============================================="
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate environment
source ~/anaconda3/bin/activate nips27

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_DIR")"

# Set distributed training parameters
# For 4 GPUs: nproc_per_node=4
# torchrun will handle device assignment automatically
echo "Starting training on $NUM_GPUS GPUs..."
echo "Start time: $(date)"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29500 \
    scripts/training/train_lora.py \
    --config "$CONFIG" 2>&1 | tee logs/training_$(basename "$OUTPUT_DIR").log

echo ""
echo "End time: $(date)"
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
