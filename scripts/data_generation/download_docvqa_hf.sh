#!/bin/bash
# Download DocVQA from HuggingFace (no registration required)
# Uses huggingface_hub to download parquet files

set -e

DATA_DIR="${1:-$HOME/datasets}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "Downloading DocVQA from HuggingFace"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo ""

# ============================================
# Download DocVQA from HuggingFace
# ============================================
echo "=== Downloading DocVQA ==="

if [ -d "DocVQA" ] && [ "$(ls -A DocVQA)" ]; then
    echo "DocVQA directory already exists, skipping..."
    ls -la DocVQA/
else
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub

    mkdir -p DocVQA
    cd DocVQA

    echo "Downloading DocVQA train/val parquet files..."
    echo "Source: https://huggingface.co/datasets/lmms-lab/DocVQA"
    echo ""

    # Download using huggingface-cli
    huggingface-cli download lmms-lab/DocVQA --repo-type dataset --local-dir .

    echo ""
    echo "DocVQA downloaded to: $DATA_DIR/DocVQA"
    echo "Contents:"
    ls -la
    cd ..
fi

echo ""
echo "=============================================="
echo "DocVQA Ready!"
echo "=============================================="
echo ""
echo "Next step: Generate mini dataset"
echo "  cd ~/cg-prm"
echo "  source ~/anaconda3/bin/activate"
echo "  conda activate nips27"
echo "  DOCVQA_DIR=~/datasets/DocVQA python scripts/generate_mini_data_docvqa.py"
echo ""
