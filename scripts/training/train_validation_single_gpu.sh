#!/bin/bash
# Single-GPU training for validation (memory optimized)
set -e

CONFIG="${1:-configs/training/validation_cg_prm.json}"
OUTPUT="${2:-outputs/validation_cg_prm}"
GPU_ID="${3:-0}"

echo "=============================================="
echo "CG-PRM Single-GPU Training (Memory Optimized)"
echo "=============================================="
echo "Config: $CONFIG"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT"
echo ""

source ~/anaconda3/bin/activate nips27
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Create backup of config
cp "$CONFIG" "${CONFIG}.backup"

# Modify config for single-GPU with smaller batch
python3 << PYTHON
import json
with open("$CONFIG", "r") as f:
    config = json.load(f)

# Reduce batch size for single GPU
config["per_device_train_batch_size"] = 1
config["gradient_accumulation_steps"] = 16  # Effective batch = 16
config["max_length"] = 2048  # Reduce from 4096

with open("$CONFIG", "w") as f:
    json.dump(config, f, indent=2)

print("Config modified for single-GPU training:")
print(f"  per_device_train_batch_size: {config['per_device_train_batch_size']}")
print(f"  gradient_accumulation_steps: {config['gradient_accumulation_steps']}")
print(f"  max_length: {config['max_length']}")
PYTHON

echo ""
echo "Starting training..."
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 scripts/training/train_lora.py --config "$CONFIG" 2>&1 | tee logs/validation_train.log

# Restore original config
mv "${CONFIG}.backup" "$CONFIG"

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT"
