# 🧪 CG-PRM Mini-Experiment - Quick Start

Use the mini-experiment to test your setup before running the full-scale experiment. **Takes ~1 hour on single GPU.**

---

## 🚀 Quick Start (One Command)

```bash
# From project root
bash run_mini_experiment.sh [GPU_ID]

# Example: Run on GPU 0
bash run_mini_experiment.sh 0
```

Or from scripts directory:
```bash
bash scripts/run_mini_experiment.sh 0
```

---

## 📋 What It Does

The mini-experiment runs the complete CG-PRM pipeline on a small dataset:

1. **Generate mini dataset** (2.5k pairs)
   - CLEVR: 1,000 clean traces → 500 F5 counterfactuals
   - DocVQA: 1,500 clean traces → 750 F5 counterfactuals

2. **Train CG-PRM model** (pairwise)
   - ~15-20 minutes on single GPU
   - Uses `configs/training/mini_cg_prm.json`

3. **Train Pointwise baseline**
   - ~15-20 minutes on single GPU
   - Uses `configs/training/mini_pointwise.json`

4. **Evaluate both models**
   - Computes AUROC, accuracy
   - Makes GO/NO-GO decision

5. **Show results**
   - Decision based on delta ≥ 0.05

---

## 📊 Expected Output

```
==============================================
CG-PRM Mini-Experiment
==============================================
GPU: 0
Working directory: /path/to/cg-prm
Start time: [timestamp]

=== Step 1: Generating mini dataset ===
Generating mini dataset in data/mini
...

=== Step 2: Training CG-PRM ===
Start: [timestamp]
[Training logs...]
End: [timestamp]

=== Step 3: Training Pointwise ===
Start: [timestamp]
[Training logs...]
End: [timestamp]

=== Step 4: Evaluating models ===
  AUROC: 0.72 [0.68, 0.76]
  Decision: GO

==============================================
RESULTS
==============================================
{
  "cg_prm": {"auroc": 0.72, ...},
  "pointwise": {"auroc": 0.65, ...},
  "delta": 0.07,
  "decision": "GO"
}

DECISION: GO
Proceed to full-scale experiment!
```

---

## 📁 Files Created

```
data/mini/
├── train_pairs.jsonl       (2,000 pairs)
├── test_pairs.jsonl        (500 pairs)
├── clevr_clean.jsonl
└── docvqa_clean.jsonl

outputs/
├── mini_cg_prm/            # CG-PRM checkpoint
└── mini_pointwise/         # Pointwise checkpoint

logs/
├── cg_prm_train.log
└── pointwise_train.log

results/
└── mini_results.json       # Main results
```

---

## ⚙️ Configuration

### Mini Dataset
- **Size:** 2,500 examples (100× smaller than full)
- **Corruptions:** F5 only (correct answer, wrong evidence)
- **Purpose:** Quick validation, not final results

### Training
- **Model:** Qwen2.5-VL-3B-Instruct
- **Batch size:** 1 per GPU
- **Gradient accumulation:** 16 steps
- **Effective batch:** 16 samples
- **Epochs:** 2
- **Learning rate:** 2e-4

### Evaluation
- **Test set:** 500 pairs
- **Metrics:** AUROC with bootstrap CI
- **Decision:** GO if delta ≥ 0.05

---

## 🧪 Step-by-Step (Manual)

If you want to run steps individually:

### Step 1: Generate Mini Data
```bash
source ~/anaconda3/bin/activate nips27
python scripts/data_generation/generate_mini_data.py
```

### Step 2: Train CG-PRM
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    scripts/training/train_lora.py \
    --config configs/training/mini_cg_prm.json
```

### Step 3: Train Pointwise
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    scripts/training/train_lora.py \
    --config configs/training/mini_pointwise.json
```

### Step 4: Evaluate
```bash
python scripts/evaluation/evaluate_mini.py \
    --cg_prm outputs/mini_cg_prm \
    --pointwise outputs/mini_pointwise \
    --test_data data/mini/test_pairs.jsonl \
    --output results/mini_results.json
```

---

## ✅ Success Criteria

### Mini-Experiment
- ✅ Both models train without errors
- ✅ Training loss decreases
- ✅ AUROC > 0.55 (random = 0.50)
- ✅ CG-PRM delta ≥ 0.02 (mini is noisy)

### Before Full-Scale
Once mini-experiment works:
1. ✅ Check training logs for errors
2. ✅ Verify checkpoints saved
3. ✅ Confirm evaluation runs
4. ✅ Ready for full-scale!

---

## ⚠️ Troubleshooting

### Out of Memory
```bash
# Reduce batch size or increase gradient accumulation
# Edit configs/training/mini_cg_prm.json:
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 32
}
```

### Stuck Processes
```bash
# Kill stuck torch processes
pkill -f torchrun
sleep 2

# Free port 29500
pkill -f 'master_port=29500'
```

### Missing Dependencies
```bash
source ~/anaconda3/bin/activate nips27
pip install -r requirements.txt
```

---

## 📈 Next Steps

After mini-experiment succeeds:

### 1. Check Results
```bash
cat results/mini_results.json | python -m json.tool
```

### 2. View Training Logs
```bash
tail -100 logs/cg_prm_train.log
tail -100 logs/pointwise_train.log
```

### 3. Monitor Training Curves
```bash
python scripts/training/monitor_training.py
```

### 4. Run Full-Scale
```bash
bash scripts/run_full_experiment.sh 4
```

---

## 🎯 Mini vs Full-Scale

| Aspect | Mini | Full-Scale |
|--------|------|------------|
| Examples | 2,500 | 100,000 |
| Corruption families | 1 (F5) | 9 (all) |
| Training pairs | 2,500 | 500,000 |
| Training time | 30 min | 4-5 days |
| Total time | 1 hour | 4-5 days |
| GPUs | 1 | 4 |
| Purpose | Testing | Final results |

---

## 📚 Documentation

- **Full experiment:** `docs/FULL_SCALE_EXPERIMENT.md`
- **Structure:** `docs/STRUCTURE_UPDATE.md`
- **Quick reference:** `QUICK_REFERENCE.md`

---

**Estimated time:** ~1 hour  
**GPU requirement:** 1× A800 80GB  
**Decision:** Test before full-scale!

**Run:** `bash run_mini_experiment.sh 0`
