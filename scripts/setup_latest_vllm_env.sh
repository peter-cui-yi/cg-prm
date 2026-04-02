#!/bin/bash
# Create fresh vLLM environment with latest compatible versions
# Usage: bash scripts/setup_latest_vllm_env.sh

set -e

ENV_NAME="nips27"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"

echo "=============================================="
echo "Setting Up Latest vLLM Environment"
echo "=============================================="
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "CUDA: $CUDA_VERSION"
echo ""



# Install latest PyTorch with CUDA 12.1
echo ""
echo "Step 2/6: Installing latest PyTorch (CUDA $CUDA_VERSION)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install latest NumPy (compatible with PyTorch)
echo ""
echo "Step 3/6: Installing NumPy..."
pip install 'numpy<2.0'

# Install latest vLLM
echo ""
echo "Step 4/6: Installing latest vLLM..."
pip install vllm

# Install additional dependencies
echo ""
echo "Step 5/6: Installing additional dependencies..."
pip install modelscope httpx tqdm transformers accelerate

# Test installation
echo ""
echo "Step 6/6: Testing installation..."
python << 'PYTHON_TEST'
import torch
import numpy

print(f"✓ PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"✓ NumPy: {numpy.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")

try:
    import vllm
    print(f"✓ vLLM: {vllm.__version__}")
    print("✓ All packages compatible!")
except Exception as e:
    print(f"✗ vLLM import failed: {e}")
    exit(1)
PYTHON_TEST

echo ""
echo "=============================================="
echo "✓ Environment Setup Complete!"
echo "=============================================="
echo ""
echo "To activate:"
echo "  source ~/anaconda3/bin/activate $ENV_NAME"
echo ""
echo "To test vLLM server:"
echo "  bash scripts/inference/launch_vllm_server.sh"
echo ""
echo "To run full experiment:"
echo "  bash scripts/run_full_experiment.sh 4"
echo ""
echo "To use in data generation:"
echo "  Edit scripts/data_generation/generate_full_data_parallel.sh"
echo "  Change: source ~/anaconda3/bin/activate vllm"
echo "  To:     source ~/anaconda3/bin/activate $ENV_NAME"
echo ""
