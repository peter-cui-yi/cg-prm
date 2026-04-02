# Full-Scale CG-PRM Experiment - Implementation Summary

This document summarizes the complete implementation of the full-scale CG-PRM experiment, extending from the mini-experiment (2.5k pairs) to full-scale (100k+ examples).

## What Was Created

### 📝 Scripts (11 new files)

1. **`scripts/download_teacher_model.sh`**
   - Downloads Qwen3VL-32B-Thinking from ModelScope
   - Usage: `bash scripts/download_teacher_model.sh [MODEL_PATH]`

2. **`scripts/launch_vllm_server.sh`**
   - Launches vLLM server for batch teacher inference
   - Supports tensor parallelism across 4 GPUs
   - Usage: `bash scripts/launch_vllm_server.sh [MODEL_PATH] [PORT] [TP_SIZE]`

3. **`scripts/vllm_batch_inference.py`**
   - Batch inference client for vLLM server
   - Features: checkpointing, async requests, progress tracking
   - Converts between CG-PRM format and vLLM format
   - Usage: `python scripts/vllm_batch_inference.py --requests REQ.jsonl --output OUT.jsonl --server-url http://localhost:8000`

4. **`scripts/generate_full_data.py`**
   - Main data generation script for full-scale experiment
   - Handles: manifests, teacher requests, inference, parsing, verification, corruptions, dataset building
   - Supports incremental execution with checkpoints
   - Usage: `python scripts/generate_full_data.py --clevr-dir DIR --docvqa-dir DIR --output-dir DIR`

5. **`scripts/evaluate_full.py`**
   - Comprehensive evaluation script
   - Computes: AUROC, accuracy, precision/recall, step-level analysis, corruption ablation
   - Usage: `python scripts/evaluate_full.py --cg_prm PATH --pointwise PATH --test_pairs PATH --output_dir DIR`

6. **`scripts/ablation_by_corruption.py`**
   - Analyzes performance by corruption family
   - Generates LaTeX tables for paper
   - Usage: `python scripts/ablation_by_corruption.py --test_pairs PATH --output OUT.json --generate-latex`

7. **`scripts/aggregate_results.py`**
   - Aggregates all evaluation results
   - Generates comprehensive summary with conclusions
   - Usage: `python scripts/aggregate_results.py --input_dir DIR --output OUT.json`

8. **`scripts/run_full_experiment.sh`**
   - Master orchestration script
   - Runs all phases: download, data generation, training, evaluation
   - Usage: `bash scripts/run_full_experiment.sh [GPU_ID] [VERIFIER_GPU]`

9. **`scripts/monitor_training.py`**
   - Monitors training progress and plots curves
   - Usage: `python scripts/monitor_training.py --log_dir logs --output plots.png`

10. **`scripts/manage_checkpoints.sh`**
    - Backup, prune, and export checkpoints
    - Usage: `bash scripts/manage_checkpoints.sh [backup|prune|export]`

11. **`scripts/download_full_datasets.sh`**
    - Downloads CLEVR and DocVQA datasets
    - Usage: `bash scripts/download_full_datasets.sh [DATA_DIR]`

### ⚙️ Configuration Files (3 new files)

1. **`configs/full_cg_prm.json`**
   - Pairwise training configuration for CG-PRM
   - Key settings: Qwen3VL-4B, LoRA r=16, 3 epochs, LR=1e-4

2. **`configs/full_pointwise.json`**
   - Pointwise baseline training configuration
   - Same hyperparameters as CG-PRM for fair comparison

3. **`configs/full_pipeline.json`**
   - Full data generation pipeline configuration
   - Specifies all paths for manifests, requests, outputs, corruptions

### 📚 Documentation (2 new files)

1. **`docs/FULL_SCALE_EXPERIMENT.md`**
   - Comprehensive guide for running full-scale experiment
   - Phase-by-phase instructions
   - Troubleshooting guide
   - Resource requirements

2. **`docs/IMPLEMENTATION_SUMMARY.md`** (this file)
   - Implementation overview
   - File structure
   - Usage examples

### 🧠 Evaluation Modules (2 new files)

1. **`src/cg_prm/evaluation/step_metrics.py`**
   - Step-level error detection metrics
   - First divergence point (t_star) analysis
   - Calibration metrics (ECE, MCE)

2. **`src/cg_prm/evaluation/corruption_ablation.py`**
   - Corruption family ablation analysis
   - Bootstrap confidence intervals
   - Best/worst family identification

### ✅ Updated Files

1. **`src/cg_prm/evaluation/__init__.py`**
   - Added exports for new evaluation modules

## File Structure

```
cg-prm/
├── scripts/
│   ├── download_teacher_model.sh          ← NEW
│   ├── launch_vllm_server.sh              ← NEW
│   ├── vllm_batch_inference.py            ← NEW
│   ├── generate_full_data.py              ← NEW
│   ├── evaluate_full.py                   ← NEW
│   ├── ablation_by_corruption.py          ← NEW
│   ├── aggregate_results.py               ← NEW
│   ├── run_full_experiment.sh             ← NEW
│   ├── monitor_training.py                ← NEW
│   ├── manage_checkpoints.sh              ← NEW
│   ├── download_full_datasets.sh          ← NEW
│   ├── ... (existing scripts)
│
├── configs/
│   ├── full_cg_prm.json                   ← NEW
│   ├── full_pointwise.json                ← NEW
│   ├── full_pipeline.json                 ← NEW
│   └── ... (existing configs)
│
├── src/cg_prm/evaluation/
│   ├── step_metrics.py                    ← NEW
│   ├── corruption_ablation.py             ← NEW
│   ├── __init__.py                        ← UPDATED
│   ├── metrics.py                         (existing)
│   └── reranking.py                       (existing)
│
├── docs/
│   ├── FULL_SCALE_EXPERIMENT.md           ← NEW
│   ├── IMPLEMENTATION_SUMMARY.md          ← NEW
│   ├── proposal.md                        (existing)
│   └── ... (existing docs)
│
└── ... (existing files)
```

## How to Run

### Option 1: Run Everything (Automated)

```bash
bash scripts/run_full_experiment.sh
```

This runs all phases sequentially:
1. Download teacher model
2. Generate full dataset (includes vLLM inference)
3. Train CG-PRM and pointwise models
4. Evaluate and aggregate results

### Option 2: Run Phase by Phase (Recommended for debugging)

```bash
# Phase 1: Download model
bash scripts/download_teacher_model.sh

# Phase 2: Generate dataset
# Step 2a: Prepare manifests and requests
python scripts/generate_full_data.py \
  --clevr-dir /path/to/CLEVR \
  --docvqa-dir /path/to/DocVQA \
  --output-dir data/full \
  --skip-teacher-inference

# Step 2b: Launch vLLM server
bash scripts/launch_vllm_server.sh

# Step 2c: Run inference
python scripts/vllm_batch_inference.py \
  --requests data/full/teacher_requests/clevr_train_requests.jsonl \
  --output data/full/teacher_outputs/clevr_train_outputs.jsonl \
  --server-url http://localhost:8000

# Step 2d: Continue data generation
python scripts/generate_full_data.py \
  --clevr-dir /path/to/CLEVR \
  --docvqa-dir /path/to/DocVQA \
  --output-dir data/full \
  --skip-manifests --skip-teacher-inference

# Phase 3: Train models
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train_lora.py \
  --config configs/full_cg_prm.json

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 scripts/train_lora.py \
  --config configs/full_pointwise.json

# Phase 4: Evaluate
python scripts/evaluate_full.py \
  --cg_prm outputs/full_cg_prm \
  --pointwise outputs/full_pointwise \
  --test_pairs data/full/training_pairs/pairwise_val.jsonl \
  --output_dir results/full \
  --step_analysis --corruption_ablation

python scripts/aggregate_results.py \
  --input_dir results/full \
  --output results/full_experiment_summary.json
```

## Key Features

### 1. Scalable Data Generation
- Processes 100k+ examples efficiently
- Checkpointing for fault tolerance
- Parallel corruption generation

### 2. Efficient Teacher Inference
- vLLM for 50-100 samples/sec throughput
- Batch processing with async requests
- Automatic retry on failures

### 3. Comprehensive Evaluation
- Standard AUROC with bootstrap CIs
- Step-level error analysis
- Per-corruption-family breakdown
- Calibration metrics

### 4. Robust Training
- LoRA fine-tuning for efficiency
- Gradient checkpointing for memory
- Cosine LR scheduling
- Multiple checkpoint saving

### 5. Monitoring and Management
- Training curve visualization
- Checkpoint backup/prune/export
- Progress tracking

## Dataset Statistics

### Training Data
- **CLEVR:** ~70k examples → ~63k verified → ~567k corrupted
- **DocVQA:** ~30k examples → ~27k verified → ~243k corrupted
- **Total pairs:** ~800k+ (before filtering)
- **Final training pairs:** ~500k (after quality filtering)

### Test Data
- **CLEVR val:** ~15k examples
- **DocVQA val:** ~5k examples
- **Total test pairs:** ~20k

### Corruption Families
- F1: wrong_region (DocVQA)
- F2: wrong_value (DocVQA)
- F3: wrong_relation
- F4: missing_step
- F5: correct_answer_wrong_evidence (main hypothesis)
- F6: hallucinated_step
- F7: order_swap
- Cross-corruptors: Multi-family corruptions
- Wrong_use: Misuse of correct evidence

## Expected Outcomes

### Main Results
| Metric | Pointwise | CG-PRM | Delta |
|--------|-----------|--------|-------|
| AUROC | ~0.65 | ~0.72 | +0.07 |
| 95% CI | [0.63, 0.67] | [0.70, 0.74] | - |
| Accuracy | ~0.62 | ~0.68 | +0.06 |

### Success Criteria
✅ **GO** if:
- Delta ≥ 0.05
- Non-overlapping 95% CIs
- CG-PRM AUROC > 0.70

### Ablation Insights
Expected best performance on:
- F5 (main hypothesis test)
- F1-F3 (direct grounding errors)

Expected worst performance on:
- F7 (subtle order errors)
- Cross-corruptors (complex errors)

## Resource Requirements

### Hardware
- **GPU:** 4× A800 80GB (teacher inference), 1-2× A800 80GB (training)
- **Storage:** ~330 GB total
- **RAM:** 64+ GB recommended

### Time
- **Model download:** 2-3 hours
- **Teacher inference:** 8-12 hours
- **Corruption generation:** 4-6 hours
- **Training (each model):** 2-3 days
- **Evaluation:** 6-12 hours
- **Total:** 7-9 days (sequential) or 4-5 days (parallelized)

## Dependencies

### Python Environment
```bash
# For data generation and training
conda activate nips27

# For teacher inference
conda activate vllm

# Required packages (in requirements.txt):
# - torch
# - transformers
# - peft
# - vllm
# - modelscope
# - scikit-learn
# - matplotlib
```

## Testing

### Test with Mini-Dataset First
```bash
# Before running full experiment, test with mini dataset
python scripts/generate_full_data.py \
  --clevr-dir /path/to/CLEVR \
  --docvqa-dir /path/to/DocVQA \
  --output-dir data/mini_test \
  --clevr-limit 100 \
  --docvqa-limit 100
```

### Verify Components
```bash
# Test vLLM server
bash scripts/launch_vllm_server.sh
curl http://localhost:8000/health

# Test training
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --config configs/mini_cg_prm.json

# Test evaluation
python scripts/evaluate_full.py \
  --cg_prm outputs/mini_cg_prm \
  --pointwise outputs/mini_pointwise \
  --test_pairs data/mini/test_pairs.jsonl \
  --output_dir results/test
```

## Next Steps

After running the full experiment:

1. **Analyze results:**
   ```bash
   cat results/full_experiment_summary.json
   ```

2. **Generate paper tables:**
   ```bash
   ls results/full/*.tex
   ```

3. **Create visualizations:**
   - Training curves: `logs/training_curves.png`
   - Corruption ablation: Custom script needed
   - AUROC curves: Add to evaluation script

4. **Write paper:**
   - Update `paper/paper_draft.md` with results
   - Include tables from `results/full/*.tex`

5. **Ablation studies:**
   - Vary corruption families
   - Vary training data size
   - Vary LoRA hyperparameters

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| vLLM CUDA error | `pip install vllm --force-reinstall` |
| OOM during inference | Reduce `--batch-size` or `--tensor-parallel-size` |
| Training loss not decreasing | Reduce LR to 5e-5 |
| Low AUROC (<0.55) | Check data quality, verify corruptions |
| Checkpoint OOM | Increase `gradient_accumulation_steps`, reduce batch size |

## Support

For issues or questions:
1. Check `docs/FULL_SCALE_EXPERIMENT.md` for detailed instructions
2. Review error logs in `logs/` directory
3. Verify file paths in configs
4. Test with mini-dataset first

## Citation

```bibtex
@article{cui2026cgprm,
  title={Counterfactual Grounding Process Reward Models for Verifiable Multimodal Reasoning},
  author={Cui, Yi and others},
  journal={arXiv preprint},
  year={2026}
}
```

---

**Implementation completed:** April 1, 2026  
**Ready for full-scale experiment:** ✅ Yes  
**Recommended next step:** Run mini-dataset test first
