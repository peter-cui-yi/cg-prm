#!/bin/bash
# Download DocVQA dataset for CG-PRM experiment
# Note: DocVQA requires free registration at https://docvqa.cs.cmu.edu/

set -e

DATA_DIR="${1:-$HOME/datasets}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "Downloading DocVQA Dataset"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo ""

# ============================================
# DocVQA Download Instructions
# ============================================
echo "=== DocVQA Download Options ==="
echo ""
echo "DocVQA requires free registration at: https://docvqa.cs.cmu.edu/"
echo ""
echo "OPTION 1: Manual Download (Recommended)"
echo "----------------------------------------"
echo "1. Register at: https://docvqa.cs.cmu.edu/"
echo "2. Download links (after login):"
echo "   - Train Images Part 1: https://docvqa.cs.cmu.edu/devkit/train_v1.0.all_part1.zip"
echo "   - Train Images Part 2: https://docvqa.cs.cmu.edu/devkit/train_v1.0.all_part2.zip"
echo "   - Train Images Part 3: https://docvqa.cs.cmu.edu/devkit/train_v1.0.all_part3.zip"
echo "   - Val Images: https://docvqa.cs.cmu.edu/devkit/val_v1.0.zip"
echo "   - Train QA: https://docvqa.cs.cmu.edu/devkit/train_v1.0.json.zip"
echo "   - Val QA: https://docvqa.cs.cmu.edu/devkit/val_v1.0.json.zip"
echo ""
echo "3. Place all zip files in: $DATA_DIR/DocVQA/"
echo "4. Run: unzip '*.zip' in the DocVQA folder"
echo ""

# Check if user has downloaded files
if [ -d "DocVQA" ] && [ -f "DocVQA/train_v1.0.json" ]; then
    echo "DocVQA already exists, skipping download..."
    ls -la DocVQA/
else
    echo "OPTION 2: Use Alternative Datasets (No Registration)"
    echo "----------------------------------------------------"
    echo ""

    # Alternative 1: TextVQA (similar to DocVQA, easier access)
    echo "Downloading TextVQA as DocVQA alternative..."
    if [ ! -d "TextVQA" ]; then
        mkdir -p TextVQA
        cd TextVQA

        # TextVQA images from Visual Genome
        echo "TextVQA uses Visual Genome images..."
        echo "Download VG images separately if needed."

        # TextVQA annotations
        wget --continue https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json -O TextVQA_0.5.1_train.json 2>/dev/null || echo "Could not download TextVQA annotations"
        wget --continue https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -O images.zip 2>/dev/null || echo "Could not download TextVQA images"

        if [ -f "images.zip" ]; then
            unzip -q images.zip
            rm -f images.zip
        fi

        cd ..
        echo "TextVQA downloaded to: $DATA_DIR/TextVQA"
    fi
    echo ""

    # Alternative 2: Use subset of existing LLaVA-150k (COCO has document-like images)
    echo "OPTION 3: Use Document-like Images from LLaVA-150k"
    echo "---------------------------------------------------"
    LLAVA_DIR="$HOME/dataset/llava-150k"
    if [ -d "$LLAVA_DIR" ]; then
        echo "Found LLaVA-150k at: $LLAVA_DIR"
        echo "We'll filter for document-like images (texts, signs, charts)"
    fi
fi

echo ""
echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo ""
echo "If you downloaded DocVQA manually:"
echo "  cd $DATA_DIR/DocVQA"
echo "  unzip '*.zip'"
echo ""
echo "Then run data generation:"
echo "  cd ~/cg-prm"
echo "  source ~/anaconda3/bin/activate"
echo "  conda activate nips27"
echo "  CLEVR_DIR=~/datasets/CLEVR_v1.0 DOCVQA_DIR=~/datasets/DocVQA python scripts/generate_mini_data_docvqa.py"
echo ""
