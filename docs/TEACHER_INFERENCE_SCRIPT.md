# 🚀 Teacher Inference Script - Multi-GPU

## Quick Start

Run teacher inference on both CLEVR and DocVQA using all 4 GPUs:

```bash
cd /hpc2hdd/home/ycui785/cg-prm
bash scripts/inference/run_teacher_inference.sh
```

## What It Does

1. **Launches vLLM server** on 4 GPUs (tensor parallelism)
2. **Processes CLEVR** (~700k examples, ~8-12 hours)
3. **Processes DocVQA** (~39k examples, ~30-60 minutes)
4. **Saves outputs** with checkpointing every 1k examples
5. **Stops server** automatically when done

## Usage

```bash
# Default (recommended)
bash scripts/inference/run_teacher_inference.sh

# Custom output directory
bash scripts/inference/run_teacher_inference.sh data/full

# Adjust batch size for throughput/memory tradeoff
bash scripts/inference/run_teacher_inference.sh data/full 64 32

# Use existing vLLM server on custom port
bash scripts/inference/run_teacher_inference.sh data/full 64 32 8000
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_DIR` | `data/full` | Output directory |
| `BATCH_SIZE` | `64` | Batch size for inference |
| `MAX_CONCURRENT` | `32` | Max concurrent requests |
| `VLLM_PORT` | `8000` | vLLM server port |
| `TP_SIZE` | `4` | Tensor parallel size (GPUs) |

## Performance

### With 4× A800 GPUs

| Dataset | Examples | Time (est.) | Throughput |
|---------|----------|-------------|------------|
| CLEVR | 700k | 8-12 hours | 50-100 samples/sec |
| DocVQA | 39k | 30-60 min | 50-100 samples/sec |
| **Total** | **739k** | **9-13 hours** | **50-100 samples/sec** |

### Optimization Tips

1. **Increase batch size** (if memory allows):
   ```bash
   bash scripts/inference/run_teacher_inference.sh data/full 128 64
   ```

2. **Decrease batch size** (if OOM):
   ```bash
   bash scripts/inference/run_teacher_inference.sh data/full 32 16
   ```

3. **Resume from checkpoint** (if interrupted):
   - Script automatically resumes from `.checkpoint.json`
   - Just re-run the same command

## Output Files

```
data/full/
├── teacher_requests/
│   ├── clevr_train_requests.jsonl   (input)
│   └── docvqa_train_requests.jsonl  (input)
├── teacher_outputs/
│   ├── clevr_train_outputs.jsonl    (output)
│   └── docvqa_train_outputs.jsonl   (output)
└── teacher_outputs/
    ├── clevr_train_outputs.checkpoint.json  (auto-save)
    └── docvqa_train_outputs.checkpoint.json (auto-save)
```

## Logs

- **vLLM server:** `logs/vllm_teacher_server.log`
- **Inference progress:** Printed to console
- **Checkpoints:** Every 1,000 examples

## Troubleshooting

### vLLM Server Won't Start

```bash
# Check if port is already in use
netstat -tulpn | grep 8000

# Kill existing process
pkill -f "vllm.entrypoints.api_server"

# Check logs
tail -100 logs/vllm_teacher_server.log
```

### Out of Memory

Reduce batch size and concurrent requests:
```bash
bash scripts/inference/run_teacher_inference.sh data/full 32 16
```

### Slow Throughput (<50 samples/sec)

Increase batch size (if GPU memory allows):
```bash
bash scripts/inference/run_teacher_inference.sh data/full 128 64
```

### Interrupted Inference

Just re-run the same command - it will resume from checkpoint:
```bash
bash scripts/inference/run_teacher_inference.sh
```

## Manual Control

### Start vLLM Server Manually

```bash
source ~/anaconda3/bin/activate nips27
export PYTHONPATH=/hpc2hdd/home/ycui785/cg-prm/src:$PYTHONPATH
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server \
    --model /hpc2hdd/home/ycui785/model/qwen3vl-4b \
    --tensor-parallel-size 4 \
    --port 8000 \
    --trust-remote-code \
    --limit-mm-per-prompt.video 0 \
    --async-scheduling
```

### Run Inference Separately

```bash
# CLEVR only
python scripts/inference/vllm_batch_inference.py \
    --requests data/full/teacher_requests/clevr_train_requests.jsonl \
    --output data/full/teacher_outputs/clevr_train_outputs.jsonl \
    --server-url http://localhost:8000 \
    --batch-size 64 \
    --max-concurrent 32

# DocVQA only
python scripts/inference/vllm_batch_inference.py \
    --requests data/full/teacher_requests/docvqa_train_requests.jsonl \
    --output data/full/teacher_outputs/docvqa_train_outputs.jsonl \
    --server-url http://localhost:8000 \
    --batch-size 64 \
    --max-concurrent 32
```

## After Completion

Continue with data generation:

```bash
python scripts/data_generation/generate_full_data.py \
    --output-dir data/full \
    --skip-manifests \
    --skip-teacher-inference
```

Then train models:

```bash
# Train CG-PRM
bash scripts/training/train_cg_prm.sh configs/training/full_cg_prm.json 4

# Train Pointwise
bash scripts/training/train_cg_prm.sh configs/training/full_pointwise.json 4
```

---

**Created:** April 1, 2026  
**Status:** ✅ Ready to use  
**GPUs:** 4× A800 with tensor parallelism
