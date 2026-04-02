# ✅ DocVQA Dataset Converted Successfully

## What Was Done

Converted HuggingFace parquet files → DocVQA JSON format

### Conversion Details

**Train Set:**
- Input: 24 parquet files (50,000 examples)
- Output: `train_v1.0.json` (19 MB)
- Location: `/hpc2hdd/home/ycui785/datasets/DocVQA/`

**Validation Set:**
- Input: 6 parquet files (5,188 examples)
- Output: `val_v1.0.json` (2.0 MB)
- Location: `/hpc2hdd/home/ycui785/datasets/DocVQA/`

### Files Created

```
/hpc2hdd/home/ycui785/datasets/DocVQA/
├── train_v1.0.json          ✓ 19 MB, 50k examples
├── val_v1.0.json            ✓ 2.0 MB, 5.2k examples
├── train_v1.0.ocr.json      - Not available in source
├── val_v1.0.ocr.json        - Not available in source
└── documents/               ✓ Image directory
```

### Conversion Script

Created: `scripts/data_generation/convert_docvqa_parquet.py`

Usage:
```bash
# Convert training set
python scripts/data_generation/convert_docvqa_parquet.py \
  --input-dir /hpc2hdd/home/ycui785/datasets/DocVQA/DocVQA \
  --output-json /hpc2hdd/home/ycui785/datasets/DocVQA/train_v1.0.json

# Convert validation set
python scripts/data_generation/convert_docvqa_parquet.py \
  --input-dir /hpc2hdd/home/ycui785/datasets/DocVQA/DocVQA_val \
  --output-json /hpc2hdd/home/ycui785/datasets/DocVQA/val_v1.0.json
```

## Updated Pipeline

Updated `scripts/data_generation/generate_full_data.py` to:
- ✅ Use converted JSON files
- ✅ Use `/documents/` directory for images
- ✅ Handle missing OCR files gracefully
- ✅ Support both CLEVR + DocVQA together

## Current Status

✅ **DocVQA:** Ready (50k train, 5.2k val)  
✅ **CLEVR:** Ready (70k train, 15k val)  
✅ **Full pipeline:** Ready to run

## How to Run

### Full-Scale Experiment (CLEVR + DocVQA)
```bash
cd /hpc2hdd/home/ycui785/cg-prm
bash scripts/run_full_experiment.sh 4
```

This will now use:
- 70k CLEVR examples
- 50k DocVQA examples
- **Total: 120k training examples**

### Expected Output
```
=== Building Manifests ===
Building CLEVR manifest...
  CLEVR train: 69,998 examples
  CLEVR val: 14,999 examples
Building DocVQA manifest...
  DocVQA train: 50,000 examples
  DocVQA val: 5,188 examples
```

## Notes

- OCR data not available in source parquet files
- Images are in `/documents/` directory (not split into train/val)
- Conversion preserves all question-answer pairs
- Format compatible with existing DocVQA pipeline

---

**Date:** April 1, 2026  
**Status:** ✅ DocVQA ready for full-scale experiment
