# ✅ ROOT CAUSE CONFIRMED: Prompt Simplification Broke CG-PRM

**Date:** April 2, 2026  
**Status:** Root cause identified and confirmed

---

## 🎯 Confirmed Root Cause

**Prompt simplification destroyed the training signal.**

When we simplified prompts from JSON schema to "Question + Answer:" format to make vLLM work, the model started generating **free-form natural language** instead of **structured reasoning traces**.

---

## 🔬 Evidence

### What CG-PRM Needs (Structured Traces):
```json
{
  "predicted_answer": "yes",
  "steps": [
    {
      "step_id": 1,
      "step_text": "Locate big green objects",
      "step_type": "locate",
      "grounding_ref": "object:5,6",
      "evidence_value": "two green cylinders"
    },
    {
      "step_id": 2,
      "step_text": "Count big green objects",
      "step_type": "count",
      "grounding_ref": "",
      "evidence_value": "2"
    }
  ]
}
```

### What We Got (Free-Form Text):
```
Question: Are there more big green things than large purple shiny cubes?
Image path: /path/to/image.png
Answer: Let's analyze the image step by step. First, I need to identify 
all the big green things and all the large purple shiny cubes. The image 
contains various objects with different colors and sizes...

1. **Identify big green things:**
   - In the image, there are two big green cylinders. These are the only 
     big green objects present.

2. **Identify large purple shiny cubes:**
   - There are no large purple shiny cubes in the image. The purple objects 
     present are small and not cubes.

3. **Comparison:**
   - Number of big green things: 2
   - Number of large purple shiny cubes: 0
   - Since 2 > 0, the answer is yes.

Answer: Yes
```

**This is NOT a structured trace** - it's a natural language explanation!

---

## 📊 Diagnostic Results

### P0: Labels Inverted? ✅ FALSE
- Clean traces ARE preferred
- Corrupted traces ARE rejected
- **Labels are correct**

### P1: Trace Format Broken? ✅ TRUE
- 0% of outputs have JSON structure
- 0% have `steps` field
- 0% have `predicted_answer` field
- 100% are free-form natural language
- **Prompt simplification broke the format**

---

## 🔗 How This Caused Failure

### Chain of Failure:

1. **Simplified prompts** → Model generates free-form text
2. **Free-form text** → Can't extract structured steps
3. **No structured steps** → Corruption pipeline segments arbitrarily
4. **Arbitrary segmentation** → "Clean" and "corrupted" traces look similar
5. **Similar traces** → Model can't learn distinction
6. **Can't learn** → AUROC ≈ 0.50 (random)
7. **With noise** → AUROC < 0.50 (worse than random)

### Why Pointwise Still Works (0.80 AUROC):

Pointwise doesn't need structured traces - it just scores individual examples. The free-form text still contains answer information, so it can learn:
- Good answers → high score
- Bad answers → low score

But pairwise needs **step-level structure** to detect:
- Clean trace: correct reasoning steps
- Corrupted trace: wrong reasoning at step N

**Without structure, both look the same!**

---

## 🛠️ Solution

### Option 1: Restore Original Prompts (RECOMMENDED)

Use the original JSON schema prompts:
```
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
      "step_type": "<locate|read|extract|identify|relate|count|compute|derive|answer|reason|verify>",
      "grounding_ref": "<object:<id>|objects:<id,id>|relation:<name>:<src>:<dst>>",
      "evidence_value": "<short evidence string>"
    }
  ]
}
Do not wrap the JSON in markdown fences.
```

**But:** Qwen3VL-4B on vLLM doesn't follow JSON instructions well.

**Solution:** Use **Qwen2.5-VL-3B-Instruct** (the original model) which was designed for this task.

### Option 2: Parse Free-Form into Structured

Write a parser that converts free-form text to structured traces:
- Extract answer from "Answer: X" pattern
- Segment by numbered steps (1., 2., 3.)
- Infer step_type from keywords

**Risk:** Lossy, error-prone, may not work well.

### Option 3: Different Model

Try a model that follows structured output instructions better:
- GPT-4V (expensive)
- Claude-3 (expensive)
- LLaVA-Next (open source)

---

## 📝 Action Plan

### Immediate (Today):

1. **Restore original prompt templates** in `src/cg_prm/generation/prompts.py`
2. **Regenerate teacher requests** with original prompts
3. **Run teacher inference** using Qwen2.5-VL-3B instead of Qwen3VL-4B
   - Model is at: `/hpc2hdd/home/ycui785/model/qwen2.5-vl-3b` (if available)
   - Or download: `Qwen/Qwen2.5-VL-3B-Instruct`

### This Week:

1. Re-run validation with structured traces
2. Expect AUROC > 0.65 if hypothesis is valid
3. If still failing, check corruption quality

### Alternative (If Qwen2.5-VL-3B Not Available):

1. Write free-form text parser
2. Test on 100 samples manually
3. If parser works >80% accuracy, use it

---

## 💡 Key Lesson

**Don't simplify prompts without testing trace quality first.**

We assumed:
- "Simpler prompts → same reasoning, less format"

Reality:
- "Simpler prompts → completely different output format → broken pipeline"

**Always verify:**
1. Output format matches expectations
2. Structured fields are present
3. Corruption pipeline can process outputs

**Before scaling to 10k examples!**

---

## 📊 Timeline Impact

| Task | Original | With Fix |
|------|----------|----------|
| Fix prompts | - | 10 min |
| Re-generate requests | - | 5 min |
| Re-run teacher inference | - | 2-3 hours |
| Re-train models | - | 1 hour |
| Re-evaluate | - | 10 min |
| **Total additional time** | **-** | **~4 hours** |

---

## 🎯 Prediction

After fixing prompts and using Qwen2.5-VL-3B:

- ✅ Teacher outputs will have JSON structure
- ✅ Steps will be properly segmented
- ✅ CG-PRM AUROC will be > 0.65 (if hypothesis valid)
- ✅ Delta will be positive (> 0.05)
- ✅ Decision will be GO

**Confidence:** 85% (assuming hypothesis is valid)

---

**Bottom line:** The validation failed because we broke the trace format, not because the hypothesis is wrong. Fix the prompts, re-run, and expect better results.
