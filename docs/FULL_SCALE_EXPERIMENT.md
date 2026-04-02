# CG-PRM Full-Scale Experiment Guide

This guide describes how to run the full-scale CG-PRM experiment, scaling from the mini-experiment (2.5k pairs) to the full dataset (100k+ examples with all corruption families).

## Overview

**Goal:** Validate that counterfactual grounding supervision (CG-PRM) improves multimodal verifier performance over pointwise baseline.

**Scale:**
- **Data:** Full CLEVR (~70k) + DocVQA (~30k) = ~100k examples
- **Corruptions:** All families (F1-F7 + cross-corruptors + wrong_use) ≈ 500k+ training pairs
- **Models:** 
  - Teacher: Qwen3VL-32B-Thinking (via vLLM)
  - Verifier: Qwen3VL-4B-Instruct (LoRA fine-tuning)
- **Evaluation:** In-domain test + step-level analysis + corruption ablation

**Estimated Time:** 7-9 days (sequential) or 4-5 days (parallelized)

## Quick Start

```bash
# Run everything (takes ~1 week)
bash scripts/run_full_experiment.sh

# Or run individual phases
```

## Phase-by-Phase Instructions

### Phase 1: Download Teacher Model

```bash
bash scripts/download_teacher_model.sh
```

This downloads Qwen3VL-32B-Thinking from ModelScope (~70GB, 2-3 hours).

**Alternative:** If you already have the model, update the path in configs.

### Phase 2: Generate Full-Scale Dataset

#### Step 2.1: Prepare Manifests and Teacher Requests

```bash
source ~/anaconda3/bin/activate nips27

python scripts/generate_full_data.py \
  --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
  --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
  --output-dir data/full \
  --skip-teacher-inference
```

This creates:
- Manifests: `data/full/manifests/`
- Teacher requests: `data/full/teacher_requests/`

#### Step 2.2: Launch vLLM Server

```bash
bash scripts/launch_vllm_server.sh \
  /hpc2hdd/home/ycui785/model/qwen3vl-32b-thinking \
  8000 \
  4
```

- Port: 8000
- Tensor parallel: 4 (uses all 4 GPUs)
- Wait ~60 seconds for server to initialize

#### Step 2.3: Run Batch Inference

```bash
source ~/anaconda3/bin/activate vllm

# CLEVR
python scripts/vllm_batch_inference.py \
  --requests data/full/teacher_requests/clevr_train_requests.jsonl \
  --output data/full/teacher_outputs/clevr_train_outputs.jsonl \
  --server-url http://localhost:8000 \
  --batch-size 64 \
  --max-concurrent 32

# DocVQA
python scripts/vllm_batch_inference.py \
  --requests data/full/teacher_requests/docvqa_train_requests.jsonl \
  --output data/full/teacher_outputs/docvqa_train_outputs.jsonl \
  --server-url http://localhost:8000 \
  --batch-size 64 \
  --max-concurrent 32
```

Expected time: 8-12 hours for 100k examples.

#### Step 2.4: Parse, Verify, and Generate Corruptions

```bash
source ~/anaconda3/bin/activate nips27

python scripts/generate_full_data.py \
  --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
  --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
  --output-dir data/full \
  --skip-manifests \
  --skip-teacher-inference
```

This:
- Parses teacher outputs → clean traces
- Verifies traces
- Generates all corruption families
- Builds pairwise and pointwise datasets

Expected time: 4-6 hours.

### Phase 3: Training

#### Train CG-PRM (Pairwise)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train_lora.py \
  --config configs/full_cg_prm.json
```

Expected time: 2-3 days.

#### Train Pointwise Baseline

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 scripts/train_lora.py \
  --config configs/full_pointwise.json
```

Expected time: 2-3 days.

**Note:** Can run both in parallel on different GPUs.

### Phase 4: Evaluation

#### Standard Evaluation

```bash
python scripts/evaluate_full.py \
  --cg_prm outputs/full_cg_prm \
  --pointwise outputs/full_pointwise \
  --test_pairs data/full/training_pairs/pairwise_val.jsonl \
  --output_dir results/full \
  --step_analysis \
  --corruption_ablation
```

#### Corruption Family Ablation

```bash
python scripts/ablation_by_corruption.py \
  --test_pairs data/full/training_pairs/pairwise_val.jsonl \
  --output results/full/corruption_ablation.json \
  --generate-latex
```

#### Aggregate Results

```bash
python scripts/aggregate_results.py \
  --input_dir results/full \
  --output results/full_experiment_summary.json \
  --generate-latex
```

## File Structure

```
cg-prm/
├── data/full/
│   ├── manifests/
│   │   ├── clevr_train.jsonl
│   │   ├── clevr_val.jsonl
│   │   ├── docvqa_train.jsonl
│   │   └── docvqa_val.jsonl
│   ├── teacher_requests/
│   │   ├── clevr_train_requests.jsonl
│   │   └── docvqa_train_requests.jsonl
│   ├── teacher_outputs/
│   │   ├── clevr_train_outputs.jsonl
│   │   └── docvqa_train_outputs.jsonl
│   ├── clean_traces/
│   │   ├── clevr_verified.jsonl
│   │   └── docvqa_verified.jsonl
│   ├── corrupted_traces/
│   │   ├── clevr_main.jsonl
│   │   ├── clevr_cross.jsonl
│   │   ├── clevr_wrong_use.jsonl
│   │   ├── docvqa_main.jsonl
│   │   └── ...
│   └── training_pairs/
│       ├── pairwise_train.jsonl
│       ├── pairwise_val.jsonl
│       ├── pointwise_train.jsonl
│       └── pointwise_val.jsonl
├── outputs/
│   ├── full_cg_prm/
│   │   └── checkpoint-*/
│   └── full_pointwise/
│       └── checkpoint-*/
├── results/full/
│   ├── full_results.json
│   ├── corruption_ablation.json
│   └── step_analysis.json
└── results/full_experiment_summary.json
```

## Configuration Files

### Training Configs

- `configs/full_cg_prm.json`: Pairwise training (CG-PRM)
- `configs/full_pointwise.json`: Pointwise baseline

**Key parameters:**
- `model_name_or_path`: Qwen3VL-4B
- `num_train_epochs`: 3
- `learning_rate`: 1e-4
- `gradient_accumulation_steps`: 8
- `lora.r`: 16, `lora.alpha`: 32

### Pipeline Config

- `configs/full_pipeline.json`: Full data generation pipeline

## Monitoring

### Training Progress

```bash
python scripts/monitor_training.py \
  --log_dir logs \
  --output logs/training_curves.png
```

### Checkpoint Management

```bash
# Backup all checkpoints
bash scripts/manage_checkpoints.sh backup

# Prune old checkpoints, keep only last 2
bash scripts/manage_checkpoints.sh prune outputs 2
```

## Expected Results

### Main Metrics

| Model | AUROC | 95% CI | Delta |
|-------|-------|--------|-------|
| Pointwise | ~0.65 | [0.63, 0.67] | - |
| CG-PRM | ~0.72 | [0.70, 0.74] | +0.07 |

**Success criteria:**
- Delta ≥ 0.05
- Non-overlapping 95% CIs
- CG-PRM AUROC > 0.70

### Corruption Family Breakdown

Best performance expected on:
- F5 (correct answer, wrong evidence) - main hypothesis
- F1-F3 (grounding errors) - direct supervision signal

Worst performance expected on:
- F7 (order swap) - subtle error type
- Cross-corruptors - multiple errors

## Troubleshooting

### vLLM Server Issues

**Problem:** CUDA symbol error
```bash
# Solution: Reinstall vLLM
source ~/anaconda3/bin/activate vllm
pip uninstall vllm -y
pip install vllm --force-reinstall
```

**Problem:** OOM during inference
```bash
# Reduce tensor parallel or batch size
bash scripts/launch_vllm_server.sh ... --tensor-parallel-size 2
python scripts/vllm_batch_inference.py ... --batch-size 32
```

### Training Issues

**Problem:** Loss not decreasing
```bash
# Check learning rate
# Reduce from 1e-4 to 5e-5 in config

# Check data quality
python -c "import json; print(json.load(open('data/full/training_pairs/pairwise_train.jsonl')))"
```

**Problem:** Out of memory
```bash
# Reduce batch size or increase gradient accumulation
# Edit configs/full_cg_prm.json:
#   "per_device_train_batch_size": 1,
#   "gradient_accumulation_steps": 16,  # Increase from 8
```

### Low AUROC

**Problem:** AUROC < 0.55
```bash
# Check corruption generation
head data/full/corrupted_traces/clevr_main.jsonl

# Verify traces are actually corrupted
# Check if model is learning (training loss should decrease)
tail logs/full_cg_prm_train.log
```

## Resource Requirements

### GPU

- **Teacher inference:** 4× A800 80GB (vLLM with TP=4)
- **Verifier training:** 1× A800 80GB per model

### Storage

- **Models:** 80 GB (teacher + verifier)
- **Datasets:** 50 GB (raw)
- **Generated data:** 170 GB (traces + pairs)
- **Checkpoints:** 30 GB
- **Total:** ~330 GB

### Time

- **Model download:** 2-3 hours
- **Teacher inference:** 8-12 hours
- **Corruption generation:** 4-6 hours
- **Training (per model):** 2-3 days
- **Evaluation:** 6-12 hours
- **Total:** 7-9 days (sequential) or 4-5 days (parallelized)

## Next Steps After Training

1. **Analyze results:** `cat results/full_experiment_summary.json`
2. **Generate plots:** Use scripts in `scripts/` directory
3. **Write paper:** Results go in paper/paper_draft.md
4. **Ablation studies:** Run additional experiments by varying:
   - Corruption families
   - Training data size
   - LoRA hyperparameters

## Citation

If you use this codebase, please cite:

```bibtex
@article{cui2026cgprm,
  title={Counterfactual Grounding Process Reward Models for Verifiable Multimodal Reasoning},
  author={Cui, Yi et al.},
  journal={arXiv preprint},
  year={2026}
}
```
