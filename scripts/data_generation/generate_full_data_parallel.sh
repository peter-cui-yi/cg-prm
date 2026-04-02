#!/bin/bash
# Multi-GPU data generation with parallel corruption generation
# Usage: bash scripts/data_generation/generate_full_data_parallel.sh [OPTIONS]

set -e

CLEVR_DIR="${CLEVR_DIR:-/hpc2hdd/home/ycui785/datasets/CLEVR}"
DOCVA_DIR="${DOCVA_DIR:-/hpc2hdd/home/ycui785/datasets/DocVQA}"
OUTPUT_DIR="${OUTPUT_DIR:-data/full}"
NUM_GPUS="${NUM_GPUS:-4}"

echo "=============================================="
echo "Full-Scale Data Generation (Multi-GPU)"
echo "=============================================="
echo "CLEVR: $CLEVR_DIR"
echo "DocVQA: $DOCVA_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPUs for inference: $NUM_GPUS"
echo ""

# Activate environment and set PYTHONPATH
source ~/anaconda3/bin/activate nips27
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src:$PYTHONPATH"

# Step 1: Generate manifests and teacher requests (CPU-only)
echo "=== Step 1: Preparing manifests and requests ==="
python scripts/data_generation/generate_full_data.py \
    --clevr-dir "$CLEVR_DIR" \
    --docvqa-dir "$DOCVA_DIR" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Manifests and requests prepared"
ls -lh "$OUTPUT_DIR/manifests/"
if [ -d "$OUTPUT_DIR/teacher_requests" ]; then
    ls -lh "$OUTPUT_DIR/teacher_requests/"
fi
echo ""

# Step 2: Launch vLLM server on all 4 GPUs
echo "=== Step 2: Launching vLLM Server ==="
bash scripts/inference/launch_vllm_server.sh \
    /hpc2hdd/home/ycui785/model/qwen3vl-32b-thinking \
    8000 \
    $NUM_GPUS &
VLLM_PID=$!

# Wait for server
echo "Waiting for vLLM server to start..."
sleep 60

# Verify server is running
if ! curl -s "http://localhost:8000/health" > /dev/null; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi
echo "✓ vLLM server running on port 8000"
echo ""

# Step 3: Run batch inference (utilizes all 4 GPUs via vLLM)
echo "=== Step 3: Running Teacher Inference ==="
echo "This will utilize all $NUM_GPUS GPUs via vLLM tensor parallelism"
echo "Estimated time: 8-12 hours for 100k examples"
echo ""

# Activate vLLM environment (try new env first, fallback to old)
if conda env list | grep -q "vllm_latest"; then
    source ~/anaconda3/bin/activate vllm_latest
else
    source ~/anaconda3/bin/activate vllm
fi

for benchmark in clevr docvqa; do
    echo "Processing ${benchmark}..."
    python scripts/inference/vllm_batch_inference.py \
        --requests "$OUTPUT_DIR/teacher_requests/${benchmark}_train_requests.jsonl" \
        --output "$OUTPUT_DIR/teacher_outputs/${benchmark}_train_outputs.jsonl" \
        --server-url http://localhost:8000 \
        --batch-size 128 \
        --max-concurrent 64 \
        --mode infer
    echo ""
done

# Stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.api_server" || true
sleep 5
echo "✓ vLLM server stopped"
echo ""

# Step 4: Generate corruptions and build datasets (CPU-only, parallelized)
echo "=== Step 4: Generating Corruptions and Building Datasets ==="
source ~/anaconda3/bin/activate nips27

python scripts/data_generation/generate_full_data.py \
    --clevr-dir "$CLEVR_DIR" \
    --docvqa-dir "$DOCVA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --skip-manifests \
    --skip-teacher-inference

echo ""
echo "=============================================="
echo "Data Generation Complete!"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR/training_pairs/" 2>/dev/null || true
echo ""
echo "Next step: Training"
echo "  bash scripts/training/train_cg_prm.sh configs/training/full_cg_prm.json $NUM_GPUS"
echo ""
