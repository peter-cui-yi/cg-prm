#!/bin/bash
# Quick Validation Experiment: 5k CLEVR + 5k DocVQA
# Total time: ~1-2 hours
set -e

CLEVR_LIMIT=5000
DOCVA_LIMIT=5000
OUTPUT_DIR="data/validation_5k"
NUM_GPUS=4

echo "=============================================="
echo "CG-PRM Quick Validation (5k + 5k)"
echo "=============================================="
echo "CLEVR: $CLEVR_LIMIT examples"
echo "DocVQA: $DOCVA_LIMIT examples"
echo "Total: $((CLEVR_LIMIT + DOCVA_LIMIT)) examples"
echo "Estimated time: 1-2 hours"
echo ""

source ~/anaconda3/bin/activate nips27
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export OMP_NUM_THREADS=1

# ===== Step 1: Generate manifests and teacher requests =====
echo "=== Step 1: Generating Manifests and Teacher Requests ==="
python scripts/data_generation/generate_full_data.py \
    --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
    --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
    --output-dir "$OUTPUT_DIR" \
    --clevr-limit $CLEVR_LIMIT \
    --docvqa-limit $DOCVA_LIMIT

echo ""
ls -lh "$OUTPUT_DIR/manifests/" "$OUTPUT_DIR/teacher_requests/"
echo ""

# ===== Step 2: Run teacher inference =====
echo "=== Step 2: Teacher Inference (4 GPUs) ==="
bash scripts/inference/run_teacher_inference.sh "$OUTPUT_DIR" 64 32 8000 $NUM_GPUS

echo ""

# ===== Step 3: Verify outputs =====
echo "=== Step 3: Verifying Outputs ==="
echo "CLEVR outputs:"
wc -l "$OUTPUT_DIR/teacher_outputs/clevr_train_outputs.jsonl"
head -1 "$OUTPUT_DIR/teacher_outputs/clevr_train_outputs.jsonl" | python -c "import json,sys; d=json.load(sys.stdin); gt=d.get('generated_text',''); print('Has response:', len(gt)>100 and 'steps' in gt.lower())"

echo ""
echo "DocVQA outputs:"
wc -l "$OUTPUT_DIR/teacher_outputs/docvqa_train_outputs.jsonl"
head -1 "$OUTPUT_DIR/teacher_outputs/docvqa_train_outputs.jsonl" | python -c "import json,sys; d=json.load(sys.stdin); gt=d.get('generated_text',''); print('Has response:', len(gt)>100 and 'steps' in gt.lower())"

echo ""

# ===== Step 4: Generate corruptions and build datasets =====
echo "=== Step 4: Generating Corruptions and Building Datasets ==="
python scripts/data_generation/generate_full_data.py \
    --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
    --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
    --output-dir "$OUTPUT_DIR" \
    --skip-manifests \
    --skip-teacher-inference

echo ""
ls -lh "$OUTPUT_DIR/training_pairs/"
echo ""

# ===== Step 5: Train models (4 GPUs) =====
echo "=== Step 5: Training Models (4 GPUs each) ==="
echo "Training CG-PRM..."
bash scripts/training/train_cg_prm.sh configs/training/validation_cg_prm.json $NUM_GPUS "outputs/validation_cg_prm"

echo ""
echo "Training Pointwise..."
bash scripts/training/train_cg_prm.sh configs/training/validation_pointwise.json $NUM_GPUS "outputs/validation_pointwise"

echo ""

# ===== Step 6: Evaluate =====
echo "=== Step 6: Evaluating Models ==="
python scripts/evaluation/evaluate_mini.py \
    --cg_prm outputs/validation_cg_prm \
    --pointwise outputs/validation_pointwise \
    --test_data "$OUTPUT_DIR/training_pairs/pairwise_val.jsonl" \
    --output results/validation_results.json

echo ""
echo "=============================================="
echo "Validation Complete!"
echo "=============================================="
echo "Results:"
cat results/validation_results.json | python -m json.tool | head -20
echo ""

