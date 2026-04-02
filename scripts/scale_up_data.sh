#!/bin/bash
# Scale up CG-PRM data generation to full experiment size
#
# Usage:
#   bash scripts/scale_up_data.sh
#
# This will:
#   1. Backup existing mini data (optional)
#   2. Regenerate CLEVR data (5000 train + 500 test)
#   3. Regenerate DocVQA data (3000 train + 300 test)
#   4. Merge into data/real_mini/ for training
#   5. Report statistics
#
# Prerequisites:
#   - CLEVR dataset at ~/datasets/CLEVR_v1.0 or ~/datasets/CLEVR/CLEVR_v1.0
#   - DocVQA dataset at ~/datasets/DocVQA (with train-*.parquet shards)
#   - Python env 'nips27' with datasets, PIL, json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Conda activation ─────────────────────────────────────────────────────────
if [[ -f "$HOME/anaconda3/bin/activate" ]]; then
  source "$HOME/anaconda3/bin/activate" nips27 2>/dev/null || \
    { source "$HOME/anaconda3/bin/activate" && conda activate nips27 2>/dev/null; } || true
fi

echo "=============================================="
echo "CG-PRM Data Scale-Up"
echo "=============================================="
echo "Repo: $REPO_ROOT"
echo "Start: $(date)"
echo ""

# ── Backup existing data (optional, comment out to skip) ─────────────────────
BACKUP_DIR="data/backup_$(date +%Y%m%d_%H%M%S)"
if [[ -d "data/mini_clevr" || -d "data/mini_docvqa" ]]; then
  echo "=== Backing up existing data to $BACKUP_DIR ==="
  mkdir -p "$BACKUP_DIR"
  [[ -d "data/mini_clevr" ]] && mv data/mini_clevr "$BACKUP_DIR/" || true
  [[ -d "data/mini_docvqa" ]] && mv data/mini_docvqa "$BACKUP_DIR/" || true
  [[ -d "data/real_mini" ]] && mv data/real_mini "$BACKUP_DIR/" || true
  echo "  Backup complete: $BACKUP_DIR"
  echo ""
fi

# ── Step 1: Generate scaled CLEVR data ───────────────────────────────────────
echo "=== Step 1: Generating CLEVR data (5000 train + 500 test) ==="
export CLEVR_DIR="${CLEVR_DIR:-$HOME/datasets/CLEVR_v1.0}"
echo "CLEVR_DIR: $CLEVR_DIR"
python scripts/generate_mini_data_clevr.py
echo ""

# ── Step 2: Generate scaled DocVQA data ──────────────────────────────────────
echo "=== Step 2: Generating DocVQA data (3000 train + 300 test) ==="
export DOCVQA_DIR="${DOCVQA_DIR:-$HOME/datasets/DocVQA}"
echo "DOCVQA_DIR: $DOCVQA_DIR"
python scripts/generate_mini_data_docvqa.py
echo ""

# ── Step 3: Merge into real_mini ─────────────────────────────────────────────
echo "=== Step 3: Merging CLEVR + DocVQA into data/real_mini/ ==="
mkdir -p data/real_mini

for split in train test; do
  clevr="data/mini_clevr/${split}_pairs.jsonl"
  docvqa="data/mini_docvqa/${split}_pairs.jsonl"
  out="data/real_mini/${split}_pairs.jsonl"
  
  if [[ ! -f "$clevr" || ! -f "$docvqa" ]]; then
    echo "ERROR: missing $clevr or $docvqa"
    exit 1
  fi
  
  cat "$clevr" "$docvqa" > "$out"
  n=$(wc -l < "$out")
  echo "  ${split}_pairs.jsonl: $n rows"
done
echo ""

# ── Step 4: Statistics ───────────────────────────────────────────────────────
echo "=============================================="
echo "Scale-Up Complete!"
echo "=============================================="
echo ""
echo "Generated data:"
echo "  CLEVR:"
echo "    Train: $(wc -l < data/mini_clevr/train_pairs.jsonl) pairs"
echo "    Test:  $(wc -l < data/mini_clevr/test_pairs.jsonl) pairs"
echo ""
echo "  DocVQA:"
echo "    Train: $(wc -l < data/mini_docvqa/train_pairs.jsonl) pairs"
echo "    Test:  $(wc -l < data/mini_docvqa/test_pairs.jsonl) pairs"
echo ""
echo "  Merged (data/real_mini/):"
echo "    Train: $(wc -l < data/real_mini/train_pairs.jsonl) pairs"
echo "    Test:  $(wc -l < data/real_mini/test_pairs.jsonl) pairs"
echo ""

# ── Quick validation ─────────────────────────────────────────────────────────
echo "Quick validation (sampling 5 pairs):"
python << 'PY'
import json, os

for name, path in [
    ('CLEVR train', 'data/mini_clevr/train_pairs.jsonl'),
    ('DocVQA train', 'data/mini_docvqa/train_pairs.jsonl'),
    ('Merged train', 'data/real_mini/train_pairs.jsonl'),
]:
    with open(path) as f:
        pairs = [json.loads(l) for l in f]
    
    valid_imgs = sum(1 for p in pairs if os.path.exists(p['positive']['image_path']))
    t_stars = [p['t_star'] for p in pairs]
    neg_has_zero = sum(1 for p in pairs if any(s['label']==0 for s in p['negative']['steps']))
    
    print(f"  {name:15} | {len(pairs):5} pairs | "
          f"images valid: {valid_imgs}/{len(pairs)} ({100*valid_imgs/len(pairs):5.1f}%) | "
          f"t_star: {min(t_stars)}-{max(t_stars)} | "
          f"neg has 0: {neg_has_zero}/{len(pairs)}")
PY

echo ""
echo "Next step: Run the scaled experiment"
echo "  bash scripts/real_mini_exp.sh 0"
echo "  # or for multi-GPU:"
echo "  bash scripts/real_mini_exp.sh 0,1,2,3"
echo ""
echo "Finished: $(date)"
echo "=============================================="
