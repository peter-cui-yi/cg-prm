#!/bin/bash
# Launch vLLM server for Qwen3-VL models (based on official vLLM docs)
# Usage: bash scripts/inference/launch_vllm_server.sh [MODEL_PATH] [PORT] [TP_SIZE]
# Docs: https://docs.vllm.com.cn/projects/recipes/en/latest/Qwen/Qwen3-VL.html

set -e

MODEL_PATH="${1:-/hpc2hdd/home/ycui785/model/qwen3vl-4b}"
PORT="${2:-8000}"
TP="${3:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=============================================="
echo "Launching vLLM Server for Qwen3-VL"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP"
echo "Environment: nips27"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Activate nips27 environment and set PYTHONPATH
source ~/anaconda3/bin/activate nips27
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Check if server is already running
if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "✓ vLLM server already running on port $PORT"
    curl -s "http://localhost:$PORT/health" | python -m json.tool
    exit 0
fi

# Launch vLLM server with Qwen3-VL specific settings (from official docs)
# Key parameters:
# - --limit-mm-per-prompt.video 0 : Disable video to save memory (image-only inference)
# - --async-scheduling : Improve throughput
# - --mm-encoder-tp-mode data : Better performance for visual encoder
echo "Starting vLLM server with Qwen3-VL optimized settings..."
echo "This may take 3-5 minutes to initialize"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1))) python -m vllm.entrypoints.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --port "$PORT" \
    --host "0.0.0.0" \
    --trust-remote-code \
    --limit-mm-per-prompt.video 0 \
    --async-scheduling \
    --mm-encoder-tp-mode data \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    2>&1 | tee logs/vllm_server.log &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"
echo ""

# Wait for server to be ready (increased timeout for large models)
echo "Waiting for server to be ready (up to 5 minutes)..."
for i in {1..150}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo ""
        echo "✓ Server is ready!"
        echo ""
        echo "=============================================="
        echo "vLLM Server Running"
        echo "=============================================="
        echo "URL: http://localhost:$PORT"
        echo "Health: http://localhost:$PORT/health"
        echo "Model: $MODEL_PATH"
        echo "GPUs: $TP"
        echo "PID: $VLLM_PID"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:$PORT/generate -d '{\"prompt\": \"Hello\", \"max_tokens\": 10}'"
        echo ""
        echo "To stop server:"
        echo "  kill $VLLM_PID"
        echo ""
        exit 0
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "ERROR: Server failed to start"
echo "Check logs/vllm_server.log for details"
kill $VLLM_PID 2>/dev/null || true
exit 1
