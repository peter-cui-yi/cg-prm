#!/bin/bash
# Download Qwen3VL-32B-Thinking from ModelScope for vLLM inference
# Usage: bash scripts/download_teacher_model.sh [MODEL_PATH]

set -e

MODEL_PATH="${1:-/hpc2hdd/home/ycui785/model/qwen3vl-32b-thinking}"

echo "=============================================="
echo "Downloading Qwen3VL-32B-Thinking"
echo "=============================================="
echo "Target path: $MODEL_PATH"
echo ""

# Activate modelscope environment
# source ~/anaconda3/bin/activate modelscope

# Create directory
mkdir -p "$(dirname "$MODEL_PATH")"

# Check if already downloaded
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "Model already exists at $MODEL_PATH"
    echo "Skipping download..."
    ls -lh "$MODEL_PATH"
else
    echo "Downloading Qwen3VL-32B-Thinking from ModelScope..."
    echo "This may take 2-3 hours depending on network speed (~70GB)"
    echo ""
    
    # Download using modelscope
    modelscope download \
        --model 'Qwen/Qwen3-VL-32B-Thinking' \
        --local_dir "$MODEL_PATH"
    
    echo ""
    echo "Download complete!"
    echo "Model location: $MODEL_PATH"
    echo ""
    echo "Verifying download..."
    ls -lh "$MODEL_PATH"/*.json 2>/dev/null || true
fi

echo ""
echo "=============================================="
echo "Teacher Model Ready!"
echo "=============================================="
echo ""
echo "Next step: Launch vLLM server"
echo "  bash scripts/launch_vllm_server.sh"
echo ""
