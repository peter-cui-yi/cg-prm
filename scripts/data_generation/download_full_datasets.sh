#!/bin/bash
# Download full-scale datasets for CG-PRM experiment
# Usage: bash scripts/download_full_datasets.sh [DATA_DIR]

set -e

DATA_DIR="${1:-/hpc2hdd/home/ycui785/datasets}"

echo "=============================================="
echo "Downloading Full-Scale Datasets"
echo "=============================================="
echo "Target directory: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# ===== CLEVR Dataset =====
echo "=== 1. CLEVR Dataset ==="

if [ -d "CLEVR/CLEVR_v1.0" ] && [ -f "CLEVR/CLEVR_v1.0/scenes.json" ]; then
    echo "CLEVR already exists at $DATA_DIR/CLEVR"
    ls -lh CLEVR/CLEVR_v1.0/
else
    echo "Downloading CLEVR v1.0..."
    
    # Download CLEVR
    if [ ! -f "CLEVR_v1.0.zip" ]; then
        wget --continue https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O CLEVR_v1.0.zip
    fi
    
    # Extract
    mkdir -p CLEVR
    cd CLEVR
    if [ ! -d "CLEVR_v1.0" ]; then
        unzip -q ../CLEVR_v1.0.zip
    fi
    cd ..
    
    echo "CLEVR downloaded to: $DATA_DIR/CLEVR/CLEVR_v1.0"
    ls -lh CLEVR/CLEVR_v1.0/
fi

echo ""

# ===== DocVQA Dataset =====
echo "=== 2. DocVQA Dataset ==="

if [ -d "DocVQA" ] && [ -f "DocVQA/train_v1.0.json" ]; then
    echo "DocVQA already exists at $DATA_DIR/DocVQA"
    ls -lh DocVQA/
else
    echo "DocVQA requires registration at https://docvqa.cs.cmu.edu/"
    echo ""
    echo "After registration, download these files:"
    echo "  1. Train Images: https://docvqa.cs.cmu.edu/devkit/train_v1.0.all_part.zip"
    echo "  2. Val Images: https://docvqa.cs.cmu.edu/devkit/val_v1.0.zip"
    echo "  3. Train QA: https://docvqa.cs.cmu.edu/devkit/train_v1.0.json.zip"
    echo "  4. Val QA: https://docvqa.cs.cmu.edu/devkit/val_v1.0.json.zip"
    echo "  5. Train OCR: https://docvqa.cs.cmu.edu/devkit/train_v1.0.ocr.json.zip"
    echo "  6. Val OCR: https://docvqa.cs.cmu.edu/devkit/val_v1.0.ocr.json.zip"
    echo ""
    echo "Then extract to: $DATA_DIR/DocVQA"
    echo ""
    echo "Expected structure:"
    echo "  DocVQA/"
    echo "    ├── train_v1.0.json"
    echo "    ├── val_v1.0.json"
    echo "    ├── train_v1.0.ocr.json"
    echo "    ├── val_v1.0.ocr.json"
    echo "    └── images/"
    echo "        ├── train/"
    echo "        └── val/"
    echo ""
    
    # Check if partially downloaded
    if [ -d "DocVQA" ]; then
        echo "DocVQA directory exists but incomplete:"
        ls -lh DocVQA/ 2>/dev/null || true
    fi
fi

echo ""

# ===== Summary =====
echo "=============================================="
echo "Download Summary"
echo "=============================================="
echo ""
echo "CLEVR:"
if [ -f "CLEVR/CLEVR_v1.0/scenes.json" ]; then
    echo "  ✓ Ready"
    echo "    Train questions: $(ls CLEVR/CLEVR_v1.0/questions/CLEVR_train_questions.json 2>/dev/null || echo 'NOT FOUND')"
    echo "    Train images: $(ls -d CLEVR/CLEVR_v1.0/images/train 2>/dev/null || echo 'NOT FOUND')"
else
    echo "  ✗ Not found at $DATA_DIR/CLEVR/CLEVR_v1.0"
fi

echo ""
echo "DocVQA:"
if [ -f "DocVQA/train_v1.0.json" ]; then
    echo "  ✓ Ready"
else
    echo "  ✗ Requires manual download (see instructions above)"
fi

echo ""
echo "Next step: Run full experiment"
echo "  bash scripts/run_full_experiment.sh"
echo ""
