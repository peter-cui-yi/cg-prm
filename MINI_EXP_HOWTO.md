# 🧪 How to Run Mini-Experiment

## Quick Start

```bash
cd /hpc2hdd/home/ycui785/cg-prm
bash scripts/run_mini_experiment.sh 0
```

**Time:** ~1 hour  
**GPU:** 1× A800 80GB

---

## What's Happening

The mini-experiment runs the complete CG-PRM pipeline:

1. ✅ **Generate mini dataset** (2 min) - 2,500 examples
2. 🔄 **Train CG-PRM** (15-20 min) - Pairwise model  
3. ⏳ **Train Pointwise** (15-20 min) - Baseline
4. 📊 **Evaluate** (2-3 min) - Compare models
5. ✅ **Results** - GO/NO-GO decision

---

## Current Status

**Training is running!** 

View progress:
```bash
tail -f logs/cg_prm_train.log
```

---

## Output Files

```
data/mini/              # Generated dataset
outputs/mini_cg_prm/    # CG-PRM checkpoint
outputs/mini_pointwise/ # Pointwise checkpoint
logs/                   # Training logs
results/mini_results.json  # Final results
```

---

## Troubleshooting

### If training fails:
```bash
# Check environment
source ~/anaconda3/bin/activate nips27
python -c "import torch; print(torch.__version__)"

# Check PYTHONPATH
export PYTHONPATH=/hpc2hdd/home/ycui785/cg-prm/src:$PYTHONPATH

# Retry
bash scripts/run_mini_experiment.sh 0
```

### If stuck processes:
```bash
pkill -f torchrun
sleep 2
bash scripts/run_mini_experiment.sh 0
```

---

## Next Steps

After mini-experiment completes:

1. Check results: `cat results/mini_results.json`
2. If GO: Run full-scale with `bash scripts/run_full_experiment.sh 4`

---

**Documentation:** See `docs/MINI_EXPERIMENT_GUIDE.md`
