# CG-PRM Full-Scale Experiment - Quick Start

## 🎯 Goal
Scale CG-PRM experiment from 2.5k pairs → 500k+ pairs using full CLEVR + DocVQA datasets.

**Hardware:** 4× A800 80GB GPUs  
**Estimated Time:** 4-5 days (with 4-GPU parallelization)

## 📋 Prerequisites

### Datasets
- **CLEVR:** `/hpc2hdd/home/ycui785/datasets/CLEVR/CLEVR_v1.0/`
- **DocVQA:** `/hpc2hdd/home/ycui785/datasets/DocVQA/`

### Models
- **Teacher:** `/hpc2hdd/home/ycui785/model/qwen3vl-32b-thinking/`
- **Verifier:** `/hpc2hdd/home/ycui785/model/qwen3vl-4b/`

### Environments
```bash
conda activate nips27  # Data generation & training
conda activate vllm    # Teacher inference
```

## 🚀 Quick Start (Recommended)

### Run Everything with 4 GPUs
```bash
bash scripts/run_full_experiment.sh 4
```

This automatically:
- ✅ Uses all 4 GPUs for vLLM teacher inference (tensor parallelism)
- ✅ Uses all 4 GPUs for training (DDP)
- ✅ Generates 100k examples with all corruptions
- ✅ Trains both CG-PRM and pointwise models
- ✅ Evaluates and aggregates results

**Total time:** ~4-5 days

## 📝 Step-by-Step

### Step 1: Download Teacher Model (2-3 hours)
```bash
bash scripts/inference/download_teacher_model.sh
```

### Step 2: Generate Dataset with 4 GPUs (12-18 hours)
```bash
# Automated (recommended)
bash scripts/data_generation/generate_full_data_parallel.sh

# Manual control
# 2a. Prepare manifests
python scripts/data_generation/generate_full_data.py \
  --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
  --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
  --output-dir data/full \
  --skip-teacher-inference

# 2b. Launch vLLM on all 4 GPUs
bash scripts/inference/launch_vllm_server.sh \
  /hpc2hdd/home/ycui785/model/qwen3vl-32b-thinking \
  8000 \
  4

# 2c. Run inference (utilizes all 4 GPUs)
source ~/anaconda3/bin/activate vllm
python scripts/inference/vllm_batch_inference.py \
  --requests data/full/teacher_requests/clevr_train_requests.jsonl \
  --output data/full/teacher_outputs/clevr_train_outputs.jsonl \
  --server-url http://localhost:8000 \
  --batch-size 128 \
  --max-concurrent 64
```

### Step 3: Train with 4 GPUs (1-2 days per model)
```bash
# Train CG-PRM (pairwise) on all 4 GPUs
bash scripts/training/train_cg_prm.sh \
  configs/training/full_cg_prm.json \
  4 \
  outputs/full_cg_prm

# Train Pointwise baseline on all 4 GPUs
bash scripts/training/train_cg_prm.sh \
  configs/training/full_pointwise.json \
  4 \
  outputs/full_pointwise
```

**Training speedup with 4 GPUs:** ~4× faster than single GPU

### Step 4: Evaluate (6-12 hours)
```bash
python scripts/evaluation/evaluate_full.py \
  --cg_prm outputs/full_cg_prm \
  --pointwise outputs/full_pointwise \
  --test_pairs data/full/training_pairs/pairwise_val.jsonl \
  --output_dir results/full \
  --step_analysis \
  --corruption_ablation

python scripts/evaluation/aggregate_results.py \
  --input_dir results/full \
  --output results/full_experiment_summary.json
```

## 📊 Expected Results

```json
{
  "cg_prm": {
    "auroc": 0.72,
    "ci_95": [0.70, 0.74]
  },
  "pointwise": {
    "auroc": 0.65,
    "ci_95": [0.63, 0.67]
  },
  "delta": 0.07,
  "decision": "GO ✅"
}
```

## 📁 Organized File Structure

```
cg-prm/
├── scripts/
│   ├── run_full_experiment.sh          # Main entry point
│   │
│   ├── inference/                      # Teacher inference
│   │   ├── download_teacher_model.sh
│   │   ├── launch_vllm_server.sh
│   │   └── vllm_batch_inference.py
│   │
│   ├── data_generation/                # Data generation
│   │   ├── generate_full_data.py
│   │   ├── generate_full_data_parallel.sh
│   │   └── download_full_datasets.sh
│   │
│   ├── training/                       # Model training
│   │   ├── train_lora.py
│   │   ├── train_cg_prm.sh
│   │   └── monitor_training.py
│   │
│   ├── evaluation/                     # Evaluation
│   │   ├── evaluate_full.py
│   │   ├── ablation_by_corruption.py
│   │   └── aggregate_results.py
│   │
│   └── utils/                          # Utilities
│       ├── manage_checkpoints.sh
│       └── ...
│
├── configs/
│   ├── training/                       # Training configs
│   │   ├── full_cg_prm.json
│   │   ├── full_pointwise.json
│   │   └── ...
│   │
│   └── pipeline/                       # Pipeline configs
│       └── full_pipeline.json
│
├── src/cg_prm/evaluation/              # Evaluation modules
│   ├── step_metrics.py
│   ├── corruption_ablation.py
│   └── ...
│
└── docs/                               # Documentation
    ├── FULL_SCALE_EXPERIMENT.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── ...
```

## ⚙️ GPU Utilization

### Teacher Inference (vLLM)
- **GPUs:** 4× A800 80GB
- **Mode:** Tensor Parallelism (TP=4)
- **Throughput:** 50-100 samples/sec
- **Time:** 8-12 hours for 100k examples

### Training (DDP)
- **GPUs:** 4× A800 80GB
- **Mode:** Distributed Data Parallel
- **Effective batch:** 2 × 4 GPUs × 4 grad_acc = 32 samples/batch
- **Time:** 1-2 days per model (vs 4-6 days on single GPU)

## 🔧 Configuration for 4 GPUs

### Training Config (`configs/training/full_cg_prm.json`)
```json
{
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 32,  // 2 × 4 GPUs × 4
  "num_train_epochs": 3,
  "learning_rate": 1e-4
}
```

### vLLM Server (`scripts/inference/launch_vllm_server.sh`)
```bash
# Uses all 4 GPUs with tensor parallelism
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server \
    --tensor-parallel-size 4 \
    ...
```

## 🧪 Test with Mini-Dataset

```bash
python scripts/data_generation/generate_full_data.py \
  --clevr-dir /path/to/CLEVR \
  --docvqa-dir /path/to/DocVQA \
  --output-dir data/test \
  --clevr-limit 100 \
  --docvqa-limit 100
```

## 📚 Documentation

- **Comprehensive guide:** `docs/FULL_SCALE_EXPERIMENT.md`
- **Implementation:** `docs/IMPLEMENTATION_SUMMARY.md`
- **Quick reference:** `QUICK_REFERENCE.md`

## ⚠️ Troubleshooting

### vLLM CUDA Error
```bash
source ~/anaconda3/bin/activate vllm
pip install vllm --force-reinstall
```

### OOM During Training
- Already optimized for 4 GPUs
- If needed, reduce `per_device_train_batch_size` to 1 in config

### Low AUROC (<0.55)
```bash
# Check training logs
tail -f logs/training_full_cg_prm.log

# Verify data quality
head data/full/training_pairs/pairwise_train.jsonl
```

## ✅ Success Criteria

- ✅ **GO**: Delta ≥ 0.05, CG-PRM AUROC > 0.70
- ⚠️ **MARGINAL**: 0.02 ≤ Delta < 0.05
- ❌ **NO-GO**: Delta < 0.02

---

**Status:** ✅ Ready with 4-GPU optimization  
**Estimated time:** 4-5 days (vs 7-9 days on single GPU)
