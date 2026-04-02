#!/bin/bash
# Run complete validation with FIXED prompt templates
# This uses simplified prompts that actually work with Qwen3VL-4B on vLLM

set -e
cd /hpc2hdd/home/ycui785/cg-prm

echo "=============================================="
echo "CG-PRM Validation (FIXED Prompts)"
echo "=============================================="
echo "CLEVR: 5,000 examples"
echo "DocVQA: 5,000 examples"
echo ""

source ~/anaconda3/bin/activate nips27
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export OMP_NUM_THREADS=1

# Clean old invalid outputs
rm -rf data/validation_5k/teacher_outputs
mkdir -p data/validation_5k/teacher_outputs

# Step 1: Teacher Inference (already done if vLLM is running)
echo "=== Step 1: Teacher Inference ==="
python scripts/inference/vllm_batch_inference.py \
  --requests data/validation_5k/teacher_requests/clevr_train_requests.jsonl \
  --output data/validation_5k/teacher_outputs/clevr_train_outputs.jsonl \
  --server-url http://localhost:8000 \
  --batch-size 64 \
  --max-concurrent 32

python scripts/inference/vllm_batch_inference.py \
  --requests data/validation_5k/teacher_requests/docvqa_train_requests.jsonl \
  --output data/validation_5k/teacher_outputs/docvqa_train_outputs.jsonl \
  --server-url http://localhost:8000 \
  --batch-size 64 \
  --max-concurrent 32

echo ""
echo "Teacher inference complete!"
wc -l data/validation_5k/teacher_outputs/*.jsonl
echo ""

# Step 2: Generate corruptions
echo "=== Step 2: Generating Corruptions ==="
python scripts/data_generation/generate_full_data.py \
  --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
  --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
  --output-dir data/validation_5k \
  --skip-manifests \
  --skip-teacher-inference

echo ""
echo "Corruption generation complete!"
ls -lh data/validation_5k/training_pairs/
echo ""

# Step 3: Train models
echo "=== Step 3: Training Models ==="
bash scripts/training/train_cg_prm.sh configs/training/validation_cg_prm.json 4 outputs/validation_cg_prm
bash scripts/training/train_cg_prm.sh configs/training/validation_pointwise.json 4 outputs/validation_pointwise

echo ""

# Step 4: Evaluate
echo "=== Step 4: Evaluating ==="
python scripts/evaluation/evaluate_mini.py \
  --cg_prm outputs/validation_cg_prm \
  --pointwise outputs/validation_pointwise \
  --test_data data/validation_5k/training_pairs/pairwise_val.jsonl \
  --output results/validation_results.json

echo ""
echo "=============================================="
echo "VALIDATION COMPLETE!"
echo "=============================================="
cat results/validation_results.json | python -m json.tool
