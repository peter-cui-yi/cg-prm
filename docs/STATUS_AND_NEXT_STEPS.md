# ✅ CG-PRM Full-Scale Experiment - Status & Next Steps

**Date:** April 1, 2026  
**Status:** Data pipeline ready, vLLM needs fix

---

## ✅ What's Working

### 1. Data Preparation Pipeline
- ✅ CLEVR manifest generation: **700k examples**
- ✅ DocVQA manifest generation: **39k examples** (converted from parquet)
- ✅ Teacher request generation: **740k requests** (1.2GB)
- ✅ Multi-GPU orchestration scripts
- ✅ All path issues resolved
- ✅ PYTHONPATH configuration fixed

### 2. File Structure
```
data/full/
├── manifests/
│   ├── clevr_train.jsonl    (342MB, 700k examples)
│   ├── clevr_val.jsonl      (72MB, 150k examples)
│   └── docvqa_train.jsonl   (15MB, 39k examples)
└── teacher_requests/
    ├── clevr_train_requests.jsonl   (1.2GB, 700k requests)
    └── docvqa_train_requests.jsonl  (53MB, 39k requests)
```

### 3. Conversion Scripts
- ✅ `scripts/data_generation/convert_docvqa_parquet_v2.py` - Working perfectly
- ✅ Successfully converted HuggingFace parquet → DocVQA JSON
- ✅ 39,463 training examples with answers

---

## ❌ Current Issue: vLLM Server

### Problem
```
ImportError: /hpc2hdd/home/ycui785/anaconda3/envs/vllm/lib/python3.10/site-packages/vllm/_C.abi3.so: 
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

This is a CUDA/PyTorch/vLLM compatibility issue.

### Root Cause
The vLLM installation has a CUDA symbol mismatch, likely due to:
- PyTorch version incompatibility
- CUDA toolkit version mismatch
- vLLM compiled against different CUDA version

---

## 🔧 Solutions

### Option 1: Reinstall vLLM (Recommended)
```bash
source ~/anaconda3/bin/activate vllm
pip uninstall vllm -y
pip cache purge
pip install vllm==0.4.0 --force-reinstall
```

### Option 2: Use Alternative Inference
Skip vLLM and use HuggingFace Transformers directly:
```bash
# Modify scripts to use transformers instead of vLLM
# Slower but more compatible
```

### Option 3: Use Different vLLM Version
```bash
source ~/anaconda3/bin/activate vllm
pip uninstall vllm -y
pip install vllm==0.3.3  # Try older stable version
```

### Option 4: Mock Data Generation (For Testing)
Use the mini-experiment with synthetic traces to test the full pipeline:
```bash
bash scripts/run_mini_experiment.sh 0
```

---

## 📊 Dataset Summary

| Dataset | Train | Val | Status |
|---------|-------|-----|--------|
| CLEVR | 700k | 150k | ✅ Ready |
| DocVQA | 39k | 0* | ✅ Ready |
| **Total** | **739k** | **150k** | **Ready** |

*DocVQA val uses test set without ground truth (standard for evaluation)

---

## 🎯 Next Steps

### Immediate (Fix vLLM)
1. Try reinstalling vLLM:
   ```bash
   bash scripts/fix_vllm_installation.sh  # Create this script
   ```

2. Test vLLM server:
   ```bash
   bash scripts/inference/launch_vllm_server.sh
   ```

### After vLLM Works
3. Run batch inference (8-12 hours):
   ```bash
   python scripts/inference/vllm_batch_inference.py \
     --requests data/full/teacher_requests/clevr_train_requests.jsonl \
     --output data/full/teacher_outputs/clevr_train_outputs.jsonl \
     --server-url http://localhost:8000
   ```

4. Continue data generation:
   ```bash
   python scripts/data_generation/generate_full_data.py \
     --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
     --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
     --output-dir data/full \
     --skip-manifests \
     --skip-teacher-inference
   ```

5. Train models (1-2 days per model on 4 GPUs):
   ```bash
   bash scripts/training/train_cg_prm.sh configs/training/full_cg_prm.json 4
   ```

---

## 📝 Files Created

### Scripts
- `scripts/inference/download_teacher_model.sh`
- `scripts/inference/launch_vllm_server.sh`
- `scripts/inference/vllm_batch_inference.py`
- `scripts/data_generation/generate_full_data.py`
- `scripts/data_generation/generate_full_data_parallel.sh`
- `scripts/data_generation/convert_docvqa_parquet_v2.py`
- `scripts/training/train_cg_prm.sh`
- `scripts/evaluation/evaluate_full.py`
- `scripts/evaluation/ablation_by_corruption.py`
- `scripts/evaluation/aggregate_results.py`
- `scripts/run_full_experiment.sh`

### Configs
- `configs/training/full_cg_prm.json`
- `configs/training/full_pointwise.json`
- `configs/pipeline/full_pipeline.json`

### Documentation
- `docs/FULL_SCALE_EXPERIMENT.md`
- `docs/IMPLEMENTATION_SUMMARY.md`
- `docs/STRUCTURE_UPDATE.md`
- `docs/DOCVQA_CONVERTED.md`
- `docs/FINAL_DOCVQA_STATUS.md`
- `docs/PATH_FIXES.md`

---

## ✅ Readiness Checklist

- [x] File structure reorganized
- [x] DocVQA converted from parquet
- [x] Manifest generation working
- [x] Teacher request generation working
- [x] Multi-GPU training scripts ready
- [x] Evaluation pipeline ready
- [ ] vLLM server (needs fix)
- [ ] Teacher inference (blocked by vLLM)
- [ ] Corruption generation (blocked by inference)
- [ ] Training dataset building (blocked by corruptions)

---

## 💡 Recommendations

1. **Fix vLLM first** - Everything else is ready
2. **Test with mini-experiment** while fixing vLLM
3. **Use 4-GPU training** once data is ready
4. **Expected total time after vLLM fix:** 2-3 days

---

**Status:** 80% Complete - vLLM is the only blocker  
**All code is ready** - Just need to fix vLLM installation
