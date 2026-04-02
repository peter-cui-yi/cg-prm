# 📊 Validation Experiment Results

**Date:** April 2, 2026  
**Dataset:** 5k CLEVR + 5k DocVQA (validation scale)  
**Status:** ❌ NO-GO

---

## 🎯 Main Results

| Model | AUROC | 95% CI | Verdict |
|-------|-------|--------|---------|
| **Pointwise Baseline** | **0.7965** | [0.7766–0.8168] | ✅ Good |
| CG-PRM (Ours) | 0.4830 | [0.4673–0.4983] | ❌ Worse than random |
| **Delta** | **-0.3134** | | ❌ Hypothesis Rejected |

---

## 📈 Interpretation

### Surprising Findings:

1. **Pointwise works well (0.80 AUROC)**
   - Baseline is correctly distinguishing good vs bad traces
   - Evaluation pipeline is working

2. **CG-PRM performs WORSE than random (< 0.50)**
   - Model is learning the WRONG signal
   - Possible causes:
     - Labels are inverted (preferred vs rejected swapped)
     - Prompt too simple - lost reasoning structure
     - Corruption generation creating ambiguous examples

3. **Large negative delta (-0.31)**
   - Not just "no improvement" - actively harmful
   - Something fundamentally wrong with approach

---

## 🔍 Root Cause Analysis

### Likely Issues:

1. **Prompt Simplification Gone Wrong**
   - We simplified prompts to "Question + Image + Answer:"
   - May have lost critical reasoning structure
   - Model generates free-form text, not structured traces

2. **Label Inversion in Pairwise Training**
   - Check if `preferred_trace` vs `rejected_trace` labels are correct
   - CG-PRM may be learning to prefer corrupted traces

3. **Trace Verification Too Lenient**
   - Only 11.4% of CLEVR traces verified
   - May be training on low-quality data

4. **Corruption Quality**
   - Corruptions may be too subtle or too obvious
   - Model learning wrong signal

---

## 📋 What Was Tested

- ✅ 5,000 CLEVR examples
- ✅ 5,000 DocVQA examples
- ✅ All corruption families (F1-F7 + cross + wrong_use)
- ✅ 9,671 pairwise training pairs
- ✅ 11,195 pointwise training pairs
- ✅ Trained on 4× A800 GPUs
- ✅ Evaluated on 1,072 test pairs

---

## 🚀 Next Steps

### Option 1: Debug Current Approach
- Check label correctness in pairwise dataset
- Analyze what CG-PRM is actually learning
- Visualize model predictions vs ground truth

### Option 2: Restore Original Prompts
- Go back to JSON schema prompts
- May need different model (Qwen3VL may not support structured output)

### Option 3: Try Different Approach
- Use rule-based trace generation instead of LLM
- Use simpler corruption families first
- Test on single corruption type (e.g., F5 only)

---

## 📁 Files Generated

- `results/validation_results.json` - Full results
- `outputs/validation_cg_prm/` - CG-PRM checkpoint
- `outputs/validation_pointwise/` - Pointwise checkpoint
- `data/validation_5k/` - All generated data

---

## 💡 Lessons Learned

1. **Prompt engineering matters** - Simplification hurt performance
2. **Need to verify labels** - Always sanity-check training data
3. **Start simple** - Test single corruption family first
4. **Mock eval is dangerous** - Always do real evaluation

---

**Conclusion:** The CG-PRM hypothesis needs reformulation. Current implementation performs worse than baseline. Recommend debugging or trying alternative approaches.
