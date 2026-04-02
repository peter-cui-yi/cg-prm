# 🚀 Latest vLLM Setup Guide

## Overview

This guide shows you how to set up the **latest vLLM** with compatible PyTorch and NumPy versions.

---

## ⚡ Quick Start (One Command)

```bash
bash scripts/setup_latest_vllm_env.sh
```

This will:
- Create a new conda environment `vllm_latest`
- Install latest PyTorch with CUDA 12.1
- Install latest vLLM
- Install all dependencies
- Test the installation

**Time:** ~15-30 minutes (depending on network speed)

---

## 📦 What Gets Installed

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime |
| PyTorch | 2.4.0+ | Deep learning framework |
| CUDA | 12.1 | GPU support |
| NumPy | 1.26.x | Numerical computing |
| vLLM | Latest | High-throughput inference |
| transformers | Latest | HuggingFace models |
| modelscope | Latest | Model downloads |

---

## 🧪 After Installation

### 1. Test vLLM Server

```bash
# Activate environment
source ~/anaconda3/bin/activate vllm_latest

# Launch server
bash scripts/inference/launch_vllm_server.sh
```

Expected output:
```
✓ Server is ready!
URL: http://localhost:8000
```

### 2. Test Batch Inference

```bash
# Small test (10 examples)
python scripts/inference/vllm_batch_inference.py \
  --requests data/test_flow/teacher_requests/clevr_train_requests.jsonl \
  --output data/test_outputs.jsonl \
  --server-url http://localhost:8000 \
  --mode infer
```

### 3. Run Full Experiment

```bash
bash scripts/run_full_experiment.sh 4
```

---

## 🔧 Manual Installation (If Script Fails)

### Step 1: Create Environment
```bash
conda create -n vllm_latest python=3.10 -y
source ~/anaconda3/bin/activate vllm_latest
```

### Step 2: Install PyTorch
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install NumPy
```bash
pip install 'numpy<2.0'
```

### Step 4: Install vLLM
```bash
pip install vllm
```

### Step 5: Install Dependencies
```bash
pip install modelscope httpx tqdm transformers accelerate
```

### Step 6: Test
```bash
python -c "import torch; import vllm; print('✓ OK')"
```

---

## 🐛 Troubleshooting

### Issue: PyTorch Download Too Slow

**Solution:** Use mirror
```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121 \
  --trusted-host download.pytorch.org
```

### Issue: vLLM Installation Fails

**Solution:** Install from source
```bash
pip uninstall vllm -y
pip install git+https://github.com/vllm-project/vllm.git
```

### Issue: CUDA Out of Memory

**Solution:** Reduce tensor parallelism or batch size
```bash
# Edit scripts/launch_vllm_server.sh
# Change TP from 4 to 2
TP="${3:-2}"

# Or reduce batch size in inference
python scripts/inference/vllm_batch_inference.py \
  --batch-size 32 \  # Was 128
  --max-concurrent 16 \  # Was 64
  ...
```

### Issue: Model Download Fails

**Solution:** Download manually with modelscope
```bash
source ~/anaconda3/bin/activate modelscope
modelscope download \
  --model 'Qwen/Qwen3VL-32B-Thinking' \
  --local_dir '/hpc2hdd/home/ycui785/model/qwen3vl-32b-thinking'
```

---

## 📊 Performance Expectations

### Hardware: 4× A800 80GB

| Task | Time | Notes |
|------|------|-------|
| Model download | 30-60 min | Qwen3VL-32B (~70GB) |
| vLLM startup | 2-5 min | Loading model |
| Inference (100k examples) | 8-12 hrs | 50-100 samples/sec |
| Memory usage | ~75GB/GPU | With TP=4 |

### Optimization Tips

1. **Tensor Parallelism:** Use all 4 GPUs
   ```bash
   bash scripts/launch_vllm_server.sh ... 4
   ```

2. **Batch Size:** Adjust based on memory
   - Max throughput: `--batch-size 128 --max-concurrent 64`
   - Memory efficient: `--batch-size 32 --max-concurrent 16`

3. **Checkpointing:** Automatic every 1000 samples
   - Resume from interruptions
   - No lost progress

---

## 🧹 Cleanup

### Remove Old Environment
```bash
conda env remove -n vllm -y
```

### Remove New Environment
```bash
conda env remove -n vllm_latest -y
```

### Clear Pip Cache
```bash
pip cache purge
```

### Free Disk Space
```bash
# Remove old model downloads
rm -rf /tmp/vllm_*
rm -rf ~/.cache/pip
```

---

## 📚 Additional Resources

- **vLLM Docs:** https://docs.vllm.ai/
- **PyTorch:** https://pytorch.org/
- **Qwen3VL:** https://modelscope.cn/models/Qwen/Qwen3VL-32B-Thinking

---

## ✅ Verification Checklist

After setup, verify:

- [ ] NumPy version < 2.0
- [ ] PyTorch version ≥ 2.4
- [ ] vLLM imports without error
- [ ] CUDA is available
- [ ] vLLM server starts successfully
- [ ] Can make inference requests
- [ ] All 4 GPUs are utilized

---

**Status:** Ready for full-scale experiment  
**Last updated:** April 1, 2026
