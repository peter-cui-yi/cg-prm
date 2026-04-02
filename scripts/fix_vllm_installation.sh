#!/bin/bash
# Fix vLLM CUDA compatibility issues
# Usage: bash scripts/fix_vllm_installation.sh

set -e

echo "=============================================="
echo "Fixing vLLM Installation"
echo "=============================================="
echo ""

# Activate vLLM environment
echo "Activating vLLM environment..."
source ~/anaconda3/bin/activate vllm

# Uninstall vLLM
echo "Uninstalling vLLM..."
pip uninstall vllm -y || true

# Clear pip cache
echo "Clearing pip cache..."
pip cache purge || true

# Check PyTorch version
echo ""
echo "Current PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Try installing compatible vLLM version
echo ""
echo "Attempting to install vLLM..."
echo "Trying vLLM 0.4.0 first..."

if pip install vllm==0.4.0 --force-reinstall --no-cache-dir 2>&1 | tee /tmp/vllm_install.log; then
    echo "✓ vLLM 0.4.0 installed successfully"
else
    echo "✗ vLLM 0.4.0 failed, trying 0.3.3..."
    if pip install vllm==0.3.3 --force-reinstall --no-cache-dir 2>&1 | tee /tmp/vllm_install.log; then
        echo "✓ vLLM 0.3.3 installed successfully"
    else
        echo "✗ vLLM 0.3.3 also failed"
        echo ""
        echo "Manual intervention required:"
        echo "1. Check PyTorch/CUDA compatibility"
        echo "2. Try: pip install vllm --no-build-isolation"
        echo "3. Or use alternative: pip install git+https://github.com/vllm-project/vllm.git"
        exit 1
    fi
fi

# Test vLLM
echo ""
echo "Testing vLLM installation..."
if python -c "import vllm; print(f'vLLM {vllm.__version__} imported successfully')" 2>&1; then
    echo "✓ vLLM working!"
else
    echo "✗ vLLM still has issues"
    echo "Check /tmp/vllm_install.log for details"
    exit 1
fi

echo ""
echo "=============================================="
echo "vLLM Installation Fixed!"
echo "=============================================="
echo ""
echo "Test the server:"
echo "  bash scripts/inference/launch_vllm_server.sh"
echo ""

