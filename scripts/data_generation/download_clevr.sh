#!/bin/bash
# Download CLEVR dataset for CG-PRM mini-experiment
# Run this on your server

set -e

DATA_DIR="${1:-$HOME/datasets}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "Downloading CLEVR Dataset"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo ""

# ============================================
# Download CLEVR (Complete set ~200MB)
# ============================================
echo "=== Downloading CLEVR ==="

if [ -d "CLEVR_v1.0" ] && [ -f "CLEVR_v1.0/scenes.json" ]; then
    echo "CLEVR already exists, skipping download..."
    ls -la CLEVR_v1.0/
else
    echo "Downloading CLEVR_v1.0.zip (~200MB)..."
    wget --continue https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O CLEVR_v1.0.zip

    echo "Extracting CLEVR..."
    unzip -q CLEVR_v1.0.zip
    rm -f CLEVR_v1.0.zip

    echo ""
    echo "CLEVR downloaded successfully!"
    echo "  Location: $DATA_DIR/CLEVR_v1.0"
    echo "  Images: $(ls CLEVR_v1.0/images/train | wc -l) train + $(ls CLEVR_v1.0/images/val | wc -l) val"
    echo "  Scenes: CLEVR_v1.0/scenes.json"
fi

echo ""
echo "=============================================="
echo "CLEVR Ready!"
echo "=============================================="
echo ""
echo "Next step: Run mini-experiment data generation"
echo "  cd ~/cg-prm"
echo "  source ~/anaconda3/bin/activate"
echo "  conda activate nips27"
echo "  python scripts/generate_mini_data_clevr.py"
echo ""
