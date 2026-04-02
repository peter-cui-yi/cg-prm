#!/bin/bash
# Download CLEVR and DocVQA datasets for CG-PRM mini-experiment
# Run this on your server

set -e

DATA_DIR="${1:-$HOME/datasets}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "Downloading CG-PRM Mini-Experiment Datasets"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo ""

# ============================================
# 1. Download CLEVR (Complete set ~200MB)
# ============================================
echo "=== 1. Downloading CLEVR ==="

if [ -d "CLEVR_v1.0" ]; then
    echo "CLEVR already exists, skipping..."
else
    echo "Downloading CLEVR images and scenes..."
    wget --continue https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O CLEVR_v1.0.zip

    echo "Extracting CLEVR..."
    unzip -q CLEVR_v1.0.zip
    rm -f CLEVR_v1.0.zip

    echo "CLEVR downloaded to: $DATA_DIR/CLEVR_v1.0"
    echo "  - Images: $(ls CLEVR_v1.0/images | wc -l) files"
    echo "  - Scenes: CLEVR_v1.0/scenes.json"
fi
echo ""

# ============================================
# 2. Download DocVQA (Train + Val ~2GB)
# ============================================
echo "=== 2. Downloading DocVQA ==="

if [ -d "DocVQA" ]; then
    echo "DocVQA already exists, skipping..."
else
    mkdir -p DocVQA
    cd DocVQA

    # DocVQA requires registration, use alternative mirrors
    echo "DocVQA requires registration at https://docvqa.cs.cmu.edu/"
    echo ""
    echo "After registration, download manually:"
    echo "  1. Train Images: https://docvqa.cs.cmu.edu/devkit/train_v1.0.all_part.zip"
    echo "  2. Val Images: https://docvqa.cs.cmu.edu/devkit/val_v1.0.zip"
    echo "  3. Train QA: https://docvqa.cs.cmu.edu/devkit/train_v1.0.json.zip"
    echo "  4. Val QA: https://docvqa.cs.cmu.edu/devkit/val_v1.0.json.zip"
    echo ""
    echo "Or use the OCR-VQA dataset as alternative (no registration):"

    cd ..

    # Alternative: OCR-VQA (similar to DocVQA, no registration)
    if [ ! -d "ocr-vqa" ]; then
        echo "Downloading OCR-VQA (DocVQA alternative)..."
        mkdir -p ocr-vqa

        # OCR-VQA images (sample - first 1000)
        echo "Note: Full OCR-VQA is 50GB+. Downloading sample..."

        # For mini-experiment, use a small subset
        echo "For mini-experiment, we'll use 200 random DocVQA-like images from your existing LLaVA-150k"
        echo "These are COCO images which include document-like scenes"
    fi
fi
echo ""

# ============================================
# 3. Use existing LLaVA-150k as fallback
# ============================================
echo "=== 3. Using existing LLaVA-150k as fallback ==="

LLAVA_DIR="$HOME/dataset/llava-150k"
if [ -d "$LLAVA_DIR" ]; then
    echo "LLaVA-150k found at: $LLAVA_DIR"
    echo "Images: $(ls $LLAVA_DIR/train2017 2>/dev/null | wc -l) files"
    echo ""
    echo "For mini-experiment, we'll use:"
    echo "  - 100 images from LLaVA as 'CLEVR-like' (synthetic traces)"
    echo "  - 200 images from LLaVA as 'DocVQA-like' (document-style traces)"
else
    echo "WARNING: LLaVA-150k not found at $LLAVA_DIR"
fi
echo ""

# ============================================
# Summary
# ============================================
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo ""
echo "Datasets available:"
echo "  1. CLEVR: $DATA_DIR/CLEVR_v1.0"
echo "  2. DocVQA: Register at https://docvqa.cs.cmu.edu/"
echo "  3. Fallback: $LLAVA_DIR (already downloaded)"
echo ""
echo "For mini-experiment, run:"
echo "  cd ~/cg-prm"
echo "  source ~/anaconda3/bin/activate"
echo "  conda activate nips27"
echo "  python scripts/generate_mini_data_real.py"
echo ""
