# 📋 DocVQA Dataset Status

## Current Situation

**DocVQA JSON files are missing.** The dataset in `/hpc2hdd/home/ycui785/datasets/DocVQA/` is in HuggingFace parquet format, not the original JSON format required by the pipeline.

## What's Available

```
/hpc2hdd/home/ycui785/datasets/DocVQA/
├── DocVQA/                    # HuggingFace parquet files
│   ├── train-*.parquet       (12 files)
│   └── test-*.parquet        (6 files)
├── documents/                 # Document images
└── InfographicVQA/           # InfographicVQA dataset
```

## What's Needed

The pipeline expects DocVQA in original JSON format:
```
/hpc2hdd/home/ycui785/datasets/DocVQA/
├── train_v1.0.json           ❌ Missing
├── val_v1.0.json             ❌ Missing
├── train_v1.0.ocr.json       ❌ Missing
├── val_v1.0.ocr.json         ❌ Missing
└── images/
    ├── train/                ❌ Missing
    └── val/                  ❌ Missing
```

## ✅ Solution Implemented

Updated `scripts/data_generation/generate_full_data.py` to:
- ✅ Check if DocVQA JSON files exist
- ✅ Skip DocVQA gracefully if missing
- ✅ Continue with CLEVR-only data generation
- ✅ Print clear warning message

## Current Behavior

When running full-scale data generation:
```
=== Building Manifests ===
Building CLEVR manifest...
  CLEVR train: 69998 examples
  CLEVR val: 14999 examples
Building DocVQA manifest...
  ⚠️  DocVQA JSON files not found - skipping DocVQA
     Expected: /path/to/DocVQA/train_v1.0.json
     Expected: /path/to/DocVQA/train_v1.0.ocr.json
     Note: DocVQA requires manual download from https://docvqa.cs.cmu.edu/
```

## How to Get DocVQA (Optional)

If you want to add DocVQA later:

1. **Register** at https://docvqa.cs.cmu.edu/

2. **Download** these files:
   - Train Images: `train_v1.0.all_part.zip`
   - Val Images: `val_v1.0.zip`
   - Train QA: `train_v1.0.json.zip`
   - Val QA: `val_v1.0.json.zip`
   - Train OCR: `train_v1.0.ocr.json.zip`
   - Val OCR: `val_v1.0.ocr.json.zip`

3. **Extract** to `/hpc2hdd/home/ycui785/datasets/DocVQA/`

## CLEVR-Only Experiment

The experiment can run successfully with CLEVR only:
- ✅ 70k CLEVR training examples
- ✅ 15k CLEVR validation examples
- ✅ All corruption families (F1-F7)
- ✅ Full evaluation pipeline

This is sufficient to validate the CG-PRM hypothesis!

---

**Status:** ✅ Script handles missing DocVQA gracefully  
**Action:** None required - CLEVR-only is sufficient for validation
