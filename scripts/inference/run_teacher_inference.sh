#!/bin/bash
# Run teacher inference on CLEVR + DocVQA datasets
# Uses all 4 GPUs (0,1,2,3) with tensor parallelism for maximum throughput
# Usage: bash scripts/inference/run_teacher_inference.sh [OUTPUT_DIR] [BATCH_SIZE] [MAX_CONCURRENT]

set -e

OUTPUT_DIR="${1:-data/full}"
BATCH_SIZE="${2:-64}"
MAX_CONCURRENT="${3:-32}"
VLLM_PORT="${4:-8000}"
TP_SIZE="${5:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "CG-PRM Teacher Inference (Multi-GPU)"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max concurrent: $MAX_CONCURRENT"
echo "vLLM port: $VLLM_PORT"
echo "Tensor parallel size: $TP_SIZE (GPUs 0,1,2,3)"
echo "Start time: $(date)"
echo ""

# Activate environment and set PYTHONPATH
source ~/anaconda3/bin/activate nips27
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export OMP_NUM_THREADS=1  # Reduce CPU contention

# Create output directory
mkdir -p "$OUTPUT_DIR/teacher_outputs"

# ===== Step 1: Check if vLLM server is already running =====
echo "=== Step 1: Checking vLLM Server ==="
if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "✓ vLLM server already running on port $VLLM_PORT"
    curl -s "http://localhost:$VLLM_PORT/health" | python -m json.tool | head -10
    USE_EXISTING_SERVER=true
else
    echo "vLLM server not running, will start it..."
    USE_EXISTING_SERVER=false
fi
echo ""

# ===== Step 2: Launch vLLM server if needed =====
if [ "$USE_EXISTING_SERVER" = false ]; then
    echo "=== Step 2: Launching vLLM Server ==="
    echo "Using all $TP_SIZE GPUs with tensor parallelism..."
    echo "This may take 3-5 minutes for initial compilation..."
    echo ""
    
    # Launch vLLM server on all 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server \
        --model /hpc2hdd/home/ycui785/model/qwen3vl-4b \
        --tensor-parallel-size $TP_SIZE \
        --port $VLLM_PORT \
        --host "0.0.0.0" \
        --trust-remote-code \
        --limit-mm-per-prompt.video 0 \
        --async-scheduling \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 256 \
        2>&1 | tee logs/vllm_teacher_server.log &
    
    VLLM_PID=$!
    echo "vLLM server started with PID: $VLLM_PID"
    echo ""
    
    # Wait for server to be ready
    echo "Waiting for vLLM server to initialize..."
    echo "First run may take 3-5 minutes (torch compilation)..."
    for i in {1..150}; do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo ""
            echo "✓ vLLM server is ready!"
            echo ""
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "  Still starting... ($((i*2)) seconds elapsed)"
            tail -5 logs/vllm_teacher_server.log 2>/dev/null | grep -E "INFO|ERROR" | tail -2 || true
        fi
        sleep 2
    done
    
    # Verify server is running
    if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "✗ ERROR: vLLM server failed to start"
        echo "Check logs/vllm_teacher_server.log for details"
        exit 1
    fi
fi
echo ""

# ===== Step 3: Run teacher inference on CLEVR =====
echo "=== Step 3: Teacher Inference on CLEVR ==="
CLEVR_REQUESTS="$OUTPUT_DIR/teacher_requests/clevr_train_requests.jsonl"
CLEVR_OUTPUT="$OUTPUT_DIR/teacher_outputs/clevr_train_outputs.jsonl"

if [ ! -f "$CLEVR_REQUESTS" ]; then
    echo "✗ ERROR: CLEVR requests not found at $CLEVR_REQUESTS"
    echo "Run data generation first:"
    echo "  python scripts/data_generation/generate_full_data.py ..."
    exit 1
fi

CLEVR_COUNT=$(wc -l < "$CLEVR_REQUESTS")
echo "Processing CLEVR dataset: $CLEVR_COUNT examples"
echo "Estimated time: $((CLEVR_COUNT / 80 / 60))-$((CLEVR_COUNT / 50 / 60)) hours"
echo ""

python scripts/inference/vllm_batch_inference.py \
    --requests "$CLEVR_REQUESTS" \
    --output "$CLEVR_OUTPUT" \
    --server-url "http://localhost:$VLLM_PORT" \
    --batch-size "$BATCH_SIZE" \
    --max-concurrent "$MAX_CONCURRENT" \
    --checkpoint-interval 1000 \
    --mode infer

echo ""
echo "✓ CLEVR inference complete!"
echo "  Output: $CLEVR_OUTPUT"
if [ -f "$CLEVR_OUTPUT" ]; then
    OUTPUT_COUNT=$(wc -l < "$CLEVR_OUTPUT")
    echo "  Examples processed: $OUTPUT_COUNT"
fi
echo ""

# ===== Step 4: Run teacher inference on DocVQA =====
echo "=== Step 4: Teacher Inference on DocVQA ==="
DOCVA_REQUESTS="$OUTPUT_DIR/teacher_requests/docvqa_train_requests.jsonl"
DOCVA_OUTPUT="$OUTPUT_DIR/teacher_outputs/docvqa_train_outputs.jsonl"

if [ ! -f "$DOCVA_REQUESTS" ]; then
    echo "✗ ERROR: DocVQA requests not found at $DOCVA_REQUESTS"
    exit 1
fi

DOCVA_COUNT=$(wc -l < "$DOCVA_REQUESTS")
echo "Processing DocVQA dataset: $DOCVA_COUNT examples"
echo "Estimated time: $((DOCVA_COUNT / 80 / 60))-$((DOCVA_COUNT / 50 / 60)) hours"
echo ""

python scripts/inference/vllm_batch_inference.py \
    --requests "$DOCVA_REQUESTS" \
    --output "$DOCVA_OUTPUT" \
    --server-url "http://localhost:$VLLM_PORT" \
    --batch-size "$BATCH_SIZE" \
    --max-concurrent "$MAX_CONCURRENT" \
    --checkpoint-interval 1000 \
    --mode infer

echo ""
echo "✓ DocVQA inference complete!"
echo "  Output: $DOCVA_OUTPUT"
if [ -f "$DOCVA_OUTPUT" ]; then
    OUTPUT_COUNT=$(wc -l < "$DOCVA_OUTPUT")
    echo "  Examples processed: $OUTPUT_COUNT"
fi
echo ""

# ===== Step 5: Summary =====
echo "=============================================="
echo "Teacher Inference Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Summary:"
echo "  CLEVR:   $CLEVR_COUNT examples → $CLEVR_OUTPUT"
echo "  DocVQA:  $DOCVA_COUNT examples → $DOCVA_OUTPUT"
echo "  Total:   $((CLEVR_COUNT + DOCVA_COUNT)) examples"
echo ""
echo "Next step: Continue data generation"
echo "  python scripts/data_generation/generate_full_data.py \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --skip-manifests \\"
echo "    --skip-teacher-inference"
echo ""

# Stop vLLM server if we started it
if [ "$USE_EXISTING_SERVER" = false ]; then
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.api_server" || true
    echo "✓ vLLM server stopped"
    echo ""
fi

echo "=============================================="
echo "All Done!"
echo "=============================================="
