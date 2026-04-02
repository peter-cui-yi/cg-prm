# 🚀 Full-Scale CG-PRM - Quick Reference Card

## One-Line Commands

```bash
# Run everything
bash scripts/run_full_experiment.sh

# Download teacher model
bash scripts/download_teacher_model.sh

# Generate dataset
python scripts/generate_full_data.py --clevr-dir /path/to/CLEVR --docvqa-dir /path/to/DocVQA --output-dir data/full --skip-teacher-inference

# Launch vLLM server
bash scripts/launch_vllm_server.sh

# Run inference
python scripts/vllm_batch_inference.py --requests data/full/teacher_requests/clevr_train_requests.jsonl --output data/full/teacher_outputs/clevr_train_outputs.jsonl --server-url http://localhost:8000

# Train CG-PRM
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train_lora.py --config configs/full_cg_prm.json

# Evaluate
python scripts/evaluate_full.py --cg_prm outputs/full_cg_prm --pointwise outputs/full_pointwise --test_pairs data/full/training_pairs/pairwise_val.jsonl --output_dir results/full
```

## File Locations

| Type | Location |
|------|----------|
| Scripts | `scripts/` (11 new files) |
| Configs | `configs/full_*.json` (3 files) |
| Docs | `docs/FULL_SCALE_*.md`, `FULL_SCALE_README.md` |
| Modules | `src/cg_prm/evaluation/*.py` (2 new) |
| Results | `results/full_experiment_summary.json` |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Teacher model | Qwen3VL-32B-Thinking |
| Verifier model | Qwen3VL-4B |
| Training data | 100k examples |
| Test data | 20k examples |
| Corruption families | 9 (F1-F7 + cross + wrong_use) |
| Training pairs | ~500k |
| Epochs | 3 |
| Learning rate | 1e-4 |
| LoRA rank | 16 |

## Expected Timeline

| Phase | Time |
|-------|------|
| Model download | 2-3 hours |
| Data generation | 12-18 hours |
| Teacher inference | 8-12 hours |
| Training (each) | 2-3 days |
| Evaluation | 6-12 hours |
| **Total** | **7-9 days** |

## Success Criteria

✅ **GO**: Delta ≥ 0.05, CG-PRM AUROC > 0.70, non-overlapping CIs  
⚠️ **MARGINAL**: 0.02 ≤ Delta < 0.05  
❌ **NO-GO**: Delta < 0.02 or CG-PRM AUROC < 0.55

## Expected Results

```
CG-PRM AUROC:    0.72 [0.70, 0.74]
Pointwise AUROC: 0.65 [0.63, 0.67]
Delta:           +0.07 ✅
Decision:        GO
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| vLLM error | `pip install vllm --force-reinstall` |
| OOM | Reduce batch size, increase gradient accumulation |
| Low AUROC | Check data quality, verify training loss |
| Missing files | Check paths in configs |

## Documentation

- 📖 **Quick start**: `FULL_SCALE_README.md`
- 📚 **Full guide**: `docs/FULL_SCALE_EXPERIMENT.md`
- 🔧 **Implementation**: `docs/IMPLEMENTATION_SUMMARY.md`
- 📊 **Completion**: `docs/COMPLETION_REPORT.md`

## Test First!

```bash
python scripts/generate_full_data.py \
  --clevr-dir /path/to/CLEVR \
  --docvqa-dir /path/to/DocVQA \
  --output-dir data/test \
  --clevr-limit 100 \
  --docvqa-limit 100
```

---

**Status**: ✅ Ready to run  
**Date**: April 1, 2026  
**Total files**: 17 new, 1 updated
