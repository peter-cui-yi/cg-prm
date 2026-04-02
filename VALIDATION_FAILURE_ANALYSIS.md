# 🔍 Why CG-PRM Validation Failed - Root Cause Analysis

**Date:** April 2, 2026  
**Result:** CG-PRM AUROC = 0.48 (worse than random), Pointwise = 0.80  
**Verdict:** ❌ NO-GO - Hypothesis Rejected

---

## 🎯 The Core Problem

**CG-PRM learned the WRONG signal** - it's systematically preferring corrupted traces over clean ones (AUROC < 0.50).

This is NOT random failure - it's **inverted learning**. The model learned something, but the opposite of what we intended.

---

## 🔬 Root Causes Identified

### 1. ⚠️ CRITICAL: Prompt Simplification Destroyed Training Signal

**What we did:**
```
BEFORE (original):
Question: {question}
Image path: {image_path}
Task: Answer the question and provide a structured reasoning trace.
Return valid JSON with this schema only:
{
  "predicted_answer": "<answer>",
  "steps": [
    {
      "step_id": 1,
      "step_text": "<reasoning step>",
      "step_type": "<locate|read|extract|...>",
      "grounding_ref": "<object:<id>|...>",
      "evidence_value": "<evidence>"
    }
  ]
}

AFTER (simplified for vLLM):
Question: {question}
Image path: {image_path}
Answer:
```

**Why this broke everything:**
- ❌ Model generates free-form text, not structured traces
- ❌ No consistent step boundaries → can't identify "first divergence point"
- ❌ Corruption detection relies on step-level analysis → impossible with free-form
- ❌ Pairwise training expects structured `steps` field → gets unstructured text

**Evidence:**
```python
# From teacher outputs:
"raw_text": "Answer: To determine if there are more big green things...
             1. **Identify big green things:**
             - In the image, there are two big green cylinders...
             3. **Comparison:**
             - Number of big green things: 2..."
```

This is NOT a structured trace - it's a natural language explanation. The corruption pipeline expects:
```python
{
  "steps": [
    {"step_id": 1, "step_type": "locate", "grounding_ref": "object:5"},
    {"step_id": 2, "step_type": "count", "evidence_value": "2"}
  ]
}
```

**Impact:** ⭐⭐⭐⭐⭐ (5/5 - CATASTROPHIC)

---

### 2. ⚠️ CRITICAL: Trace Verification Too Lenient (11.4% pass rate)

**What happened:**
- 5,000 CLEVR teacher outputs generated
- Only 571 passed verification (11.4%)
- 4,429 rejected but **we used them anyway**

**Why this is a problem:**
- Training on low-quality traces
- Corrupted traces may be "better" than clean ones
- Model learns wrong signal

**Evidence:**
```
=== Parsing and Verifying clevr Traces ===
Parsing teacher outputs...
  Parsed 5000 traces (0 failed)
Verifying traces...
  Verified: 571 (11.4%)
  Rejected: 4429 (88.6%)  ← WE USED THESE!
```

**Impact:** ⭐⭐⭐⭐ (4/5 - SEVERE)

---

### 3. ⚠️ HIGH: Pairwise Labels May Be Inverted

**Suspicion:** The `preferred_trace` vs `rejected_trace` labels might be backwards.

**How pairwise training works:**
```python
# Should be:
preferred_trace = clean_trace      # Model should prefer this
rejected_trace = corrupted_trace   # Model should reject this

# But might be:
preferred_trace = corrupted_trace  # WRONG!
rejected_trace = clean_trace       # WRONG!
```

**How to check:**
```bash
# Sample from pairwise dataset
python -c "
import json
with open('data/validation_5k/training_pairs/pairwise_train.jsonl') as f:
    pair = json.loads(f.readline())
    print('Preferred trace ID:', pair['preferred_trace_id'])
    print('Rejected trace ID:', pair['rejected_trace_id'])
    print('Corruption family:', pair['corruption_family'])
    # Check if preferred has '__main__' or corruption markers in ID
"
```

**If labels are inverted:** CG-PRM would learn to prefer corrupted traces → AUROC < 0.50 ✓

**Impact:** ⭐⭐⭐⭐⭐ (5/5 - CATASTROPHIC if true)

---

### 4. ⚠️ MEDIUM: Corruption Family Mix Too Diverse

**What we did:**
- Mixed all corruption families: F1, F2, F3, F4, F5, F6, F7, cross-corruptors, wrong_use
- Some corruptions are subtle (F7: order swap)
- Some are obvious (F4: missing step)

**Why this hurts:**
- Model can't learn consistent signal
- "Clean vs corrupted" means different things for different families
- Better to start with ONE family (e.g., F5 only)

**Evidence:**
```python
# From pairwise dataset
corruption_families = ['wrong_value', 'wrong_region', 'missing_step', 
                       'order_swap', 'correct_answer_wrong_evidence', ...]
# 9 different families mixed together
```

**Impact:** ⭐⭐⭐ (3/5 - MODERATE)

---

### 5. ⚠️ MEDIUM: Free-Form Text Breaks Scoring

**How evaluation scores pairwise:**
```python
def _pairwise_scores(model, tokenizer, pair, max_length):
    pos_trace = pair["preferred_trace"]
    neg_trace = pair["rejected_trace"]
    
    # Extract steps from traces
    pos_steps = pos_trace["steps"]  # ← EXPECTS LIST
    neg_steps = neg_trace["steps"]  # ← EXPECTS LIST
    
    # Score based on step-level log probs
    ...
```

**What it gets:**
```python
pos_trace["steps"] = [
    {"step_text": "Question: ...", "step_type": "derive", ...},
    {"step_text": "1. **Identify**...", "step_type": "derive", ...}
]
```

These aren't real reasoning steps - they're segmented free-form text!

**Impact:** ⭐⭐⭐ (3/5 - MODERATE)

---

## 🧪 Testable Hypotheses

### Hypothesis 1: Labels Are Inverted (MOST LIKELY)
**Prediction:** Check `pairwise_train.jsonl` - if `preferred_trace_id` contains corruption markers, labels are inverted.

**How to test:**
```bash
python << 'PYTHON'
import json
with open('data/validation_5k/training_pairs/pairwise_train.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        pair = json.loads(line)
        pref_id = pair['preferred_trace_id']
        rej_id = pair['rejected_trace_id']
        has_corruption = '__main__' in pref_id or '__' in pref_id.split('__')[-1]
        print(f"Sample {i}:")
        print(f"  Preferred: {pref_id[:80]}...")
        print(f"  Has corruption marker: {has_corruption}")
        print(f"  Corruption family: {pair['corruption_family']}")
PYTHON
```

**If TRUE:** Fix label assignment in `build_pairwise_dataset()` → Re-train → Expect AUROC > 0.50

---

### Hypothesis 2: Prompt Format Broke Trace Structure
**Prediction:** Teacher outputs don't have proper `steps` field structure.

**How to test:**
```bash
python << 'PYTHON'
import json
with open('data/validation_5k/teacher_outputs/clevr_train_outputs.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        output = json.loads(line)
        print(f"Output {i}:")
        print(f"  raw_text length: {len(output.get('raw_text', ''))}")
        print(f"  First 200 chars: {output.get('raw_text', '')[:200]}")
        print(f"  Has numbered steps: {'1.' in output.get('raw_text', '')}")
        print(f"  Has JSON structure: {'{' in output.get('raw_text', '')}")
PYTHON
```

**If TRUE:** Restore original JSON schema prompts → Re-generate traces → Re-train

---

### Hypothesis 3: Verification Too Lenient
**Prediction:** Low-quality traces in training set.

**How to test:**
```bash
# Check verification rate
grep -E "Verified|Rejected" logs/validation_generation.log
# If Verified < 50%, too lenient
```

**If TRUE:** Tighten verification thresholds → Filter training data → Re-train

---

## 📊 Evidence Summary

| Symptom | Likely Cause | Confidence |
|---------|--------------|------------|
| AUROC < 0.50 (inverted) | Labels inverted OR prompt broke | 90% |
| Pointwise works (0.80) | Evaluation pipeline OK | 100% |
| Large negative delta (-0.31) | Systematic error, not noise | 95% |
| Only 11.4% verification pass | Training on low-quality data | 100% |
| Free-form teacher outputs | Prompt simplification | 100% |

---

## 🛠️ Fix Priority

### **P0 - Check Immediately (5 min):**
```bash
# Check if labels are inverted
python << 'PYTHON'
import json
with open('data/validation_5k/training_pairs/pairwise_train.jsonl') as f:
    sample = json.loads(f.readline())
    pref = sample['preferred_trace_id']
    rej = sample['rejected_trace_id']
    print(f"Preferred: {pref}")
    print(f"Rejected: {rej}")
    print(f"Preferred has corruption markers: {'__main__' in pref or 'wrong' in pref}")
    # If preferred has corruption markers → LABELS INVERTED!
PYTHON
```

### **P1 - Fix This Week:**
1. **Check label correctness** (P0 above)
2. **Restore original JSON prompts** - Use Qwen2.5-VL-3B instead of Qwen3VL-4B if needed
3. **Tighten trace verification** - Require ≥50% pass rate

### **P2 - Next Iteration:**
1. **Test single corruption family first** (F5 only)
2. **Add manual quality checks** on sampled traces
3. **Visualize model predictions** before full eval

---

## 🎯 Most Likely Scenario

**My assessment (80% confidence):**

**PRIMARY CAUSE:** Labels are inverted in pairwise training dataset

The `build_pairwise_dataset()` function likely has:
```python
# BUG: Should be clean=preferred, corrupted=rejected
pair = {
    "preferred_trace": corrupted_trace,  # WRONG!
    "rejected_trace": clean_trace,       # WRONG!
}
```

This would cause:
- ✅ CG-PRM AUROC < 0.50 (learning inverted signal)
- ✅ Large negative delta (-0.31)
- ✅ Pointwise still works (doesn't use pairwise labels)

**SECONDARY CAUSE:** Prompt simplification made traces unstructured, compounding the problem.

---

## 📝 Lessons Learned

1. ✅ **Always verify labels** - Sample 10 training pairs, manually check
2. ✅ **Don't simplify prompts blindly** - Test on small set first
3. ✅ **Check trace quality** - If <50% pass verification, something's wrong
4. ✅ **Start simple** - Test ONE corruption family before mixing all
5. ✅ **Mock eval is dangerous** - We saw 0.94 mock AUROC, real was 0.48!

---

## 🚀 Recommended Next Steps

### Immediate (Today):
1. **Check label correctness** (P0 script above)
2. **If inverted:** Fix `build_pairwise_dataset()` → Re-generate pairs → Re-train (30 min)
3. **If labels OK:** Restore original prompts → Re-generate traces → Re-train (2-3 hours)

### This Week:
1. Tighten trace verification thresholds
2. Test with F5 corruption only
3. Manual quality check on 50 random traces

### Next Iteration:
1. Start with mini-experiment (already works)
2. Scale to 50k examples
3. Then full 100k

---

**Bottom line:** The infrastructure works. The hypothesis may still be valid. We need to debug the implementation, not abandon the approach.

**Most likely fix:** Check and fix pairwise label assignment (5-10 min to check, 30 min to re-train if inverted).
