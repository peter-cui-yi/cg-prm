# ✅ Teacher Inference Script Ready!

## Created Files

1. **`scripts/inference/run_teacher_inference.sh`** - Main inference script
2. **`docs/TEACHER_INFERENCE_SCRIPT.md`** - Documentation

## Quick Start

Run teacher inference on **CLEVR + DocVQA** using **all 4 GPUs**:

```bash
cd /hpc2hdd/home/ycui785/cg-prm
bash scripts/inference/run_teacher_inference.sh
```

That's it! The script will:

1. ✅ Start vLLM server on 4 GPUs (tensor parallelism)
2. ✅ Process CLEVR (700k examples, ~8-12 hours)
3. ✅ Process DocVQA (39k examples, ~30-60 min)
4. ✅ Auto-save checkpoints every 1k examples
5. ✅ Stop server when done

## GPU Utilization

Uses **all 4 A800 GPUs** with tensor parallelism:
- **CUDA_VISIBLE_DEVICES=0,1,2,3**
- **Tensor parallel size: 4**
- **Memory per GPU:** ~20 GB
- **Throughput:** 50-100 samples/sec

## Estimated Time

| Dataset | Examples | Time |
|---------|----------|------|
| CLEVR | 700,000 | 8-12 hours |
| DocVQA | 39,000 | 30-60 min |
| **Total** | **739,000** | **9-13 hours** |

## Monitoring

The script prints progress:
```
=== Step 3: Teacher Inference on CLEVR ===
Processing CLEVR dataset: 699989 examples
Estimated time: 8-12 hours

Batches: 45%|████▌     | 450/1000 [02:15<01:30,  6.11it/s]
Checkpoint saved: 45000/699989
```

## Customization

```bash
# Higher throughput (if GPU memory allows)
bash scripts/inference/run_teacher_inference.sh data/full 128 64

# Lower memory usage
bash scripts/inference/run_teacher_inference.sh data/full 32 16

# Custom output directory
bash scripts/inference/run_teacher_inference.sh data/my_experiment
```

## After Completion

Continue with corruption generation and training:

```bash
# Generate corruptions and build datasets
python scripts/data_generation/generate_full_data.py \
    --output-dir data/full \
    --skip-manifests \
    --skip-teacher-inference

# Train models (uses all 4 GPUs)
bash scripts/training/train_cg_prm.sh configs/training/full_cg_prm.json 4
bash scripts/training/train_cg_prm.sh configs/training/full_pointwise.json 4
```

## If Interrupted

Just re-run the same command - it **automatically resumes** from checkpoint:

```bash
bash scripts/inference/run_teacher_inference.sh
```

## Logs

- **vLLM server:** `logs/vllm_teacher_server.log`
- **Checkpoints:** `data/full/teacher_outputs/*.checkpoint.json`
- **Outputs:** `data/full/teacher_outputs/*.jsonl`

## Test First (Optional)

Test with a small subset first:

```bash
# Generate test requests (100 examples each)
python scripts/data_generation/generate_full_data.py \
    --clevr-dir /hpc2hdd/home/ycui785/datasets/CLEVR \
    --docvqa-dir /hpc2hdd/home/ycui785/datasets/DocVQA \
    --output-dir data/test \
    --clevr-limit 100 \
    --docvqa-limit 100 \
    --skip-teacher-inference

# Run inference on test set
bash scripts/inference/run_teacher_inference.sh data/test
```

---

**Status:** ✅ Ready to run  
**GPUs:** 4× A800 (tensor parallelism)  
**Total examples:** 739,000  
**Estimated time:** 9-13 hours  

**Run:** `bash scripts/inference/run_teacher_inference.sh`
