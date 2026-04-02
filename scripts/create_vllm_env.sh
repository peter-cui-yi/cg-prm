#!/bin/bash
# Create fresh vLLM conda environment with compatible versions
set -e

ENV_NAME="vllm_fixed"
PYTHON_VERSION="3.10"

echo "=============================================="
echo "Creating Fresh vLLM Environment"
echo "=============================================="
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo ""

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists."
    echo "Remove it first with: conda env remove -n $ENV_NAME"
    exit 0
fi

# Create environment
echo "Creating conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate
source ~/anaconda3/bin/activate $ENV_NAME

# Install PyTorch first (compatible version)
echo "Installing PyTorch 2.3.0..."
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install NumPy (compatible version)
echo "Installing NumPy 1.26.4..."
pip install numpy==1.26.4

# Install vLLM (compatible with PyTorch 2.3)
echo "Installing vLLM 0.4.2..."
pip install vllm==0.4.2

# Install other dependencies
echo "Installing additional dependencies..."
pip install modelscope httpx tqdm

# Test installation
echo ""
echo "Testing installation..."
python -c "
import torch
import numpy
import vllm
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ vLLM: {vllm.__version__}')
print('✓ All compatible!')
"

echo ""
echo "=============================================="
echo "✓ Environment Created Successfully!"
echo "=============================================="
echo ""
echo "To use:"
echo "  source ~/anaconda3/bin/activate $ENV_NAME"
echo ""
echo "To launch vLLM server:"
echo "  bash scripts/launch_vllm_server.sh"
echo ""

