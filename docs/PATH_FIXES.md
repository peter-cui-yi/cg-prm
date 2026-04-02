# ✅ Path Issues Fixed

## Problems Fixed

### 1. PYTHONPATH Not Set
**Issue:** Scripts couldn't find `cg_prm` module  
**Fix:** Added `export PYTHONPATH` in:
- `scripts/run_mini_experiment.sh`
- `scripts/run_full_experiment.sh`
- `scripts/data_generation/generate_full_data_parallel.sh`

### 2. CLEVR Scene Graphs Path
**Issue:** Wrong path to scene graphs  
**Old:** `/datasets/CLEVR/CLEVR_v1.0/scenes.json`  
**New:** `/datasets/CLEVR/CLEVR_v1.0/scenes/CLEVR_train_scenes.json`  

**Fixed files:**
- `scripts/data_generation/generate_full_data.py` (lines 69, 80)
- `configs/pipeline/full_pipeline.json` (line 22)

### 3. Script Directory Paths
**Issue:** Scripts organized in subdirectories but paths not updated  
**Fix:** Updated all paths to use new structure:
- `scripts/data_generation/` - Data generation scripts
- `scripts/training/` - Training scripts
- `scripts/evaluation/` - Evaluation scripts
- `scripts/inference/` - Inference scripts

## Current Status

✅ **Mini-experiment:** Working - Training in progress  
✅ **Full-scale data generation:** Working - Manifests being created

## How to Run

### Mini-Experiment
```bash
cd /hpc2hdd/home/ycui785/cg-prm
bash scripts/run_mini_experiment.sh 0
```

### Full-Scale Experiment
```bash
cd /hpc2hdd/home/ycui785/cg-prm
bash scripts/run_full_experiment.sh 4
```

## Verified Paths

All scripts now correctly:
1. ✅ Activate `nips27` conda environment
2. ✅ Set PYTHONPATH to include `src/`
3. ✅ Use correct absolute paths for datasets
4. ✅ Reference scripts from project root

---

**Date:** April 1, 2026  
**Status:** ✅ All path issues resolved
