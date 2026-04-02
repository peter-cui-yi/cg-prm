#!/bin/bash
# Fix vLLM environment - NumPy and PyTorch version issues
set -e

echo "=============================================="
echo "Fixing vLLM Environment"
echo "=============================================="
echo ""

# Activate vLLM environment
source ~/anaconda3/bin/activate vllm

echo "Current versions:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: not found"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: not found"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: not found"
echo ""

# Downgrade NumPy to 1.x (compatible with vLLM)
echo "Step 1: Downgrading NumPy to 1.26.4 (compatible with vLLM)..."
pip install 'numpy<2.0' --force-reinstall --no-cache-dir
echo ""

# Upgrade PyTorch to 2.4+ (required by vLLM)
echo "Step 2: Upgrading PyTorch to 2.4.0..."
pip install torch==2.4.0 torchvision==0.19.0 --force-reinstall --no-cache-dir
echo ""

# Reinstall vLLM with compatible versions
echo "Step 3: Reinstalling vLLM..."
pip uninstall vllm -y || true
pip install vllm==0.5.0 --no-cache-dir
echo ""

# Test the installation
echo "Step 4: Testing installation..."
python -c "
import numpy
import torch
import vllm
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ vLLM: {vllm.__version__}')
print('✓ All imports successful!')
"

echo ""
echo "=============================================="
echo "✓ vLLM Environment Fixed!"
echo "=============================================="
echo ""
echo "Test the server:"
echo "  bash scripts/inference/launch_vllm_server.sh"
echo ""

