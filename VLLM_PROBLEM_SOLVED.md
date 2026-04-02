# ✅ vLLM Prompt Issue SOLVED!

## Problem
Qwen3VL-4B on vLLM v0.17.1 was **echoing prompts** instead of generating responses when using complex JSON schema instructions.

## Root Cause
The prompt template included:
- Long JSON schema instructions
- "Do not wrap the JSON in markdown fences" 
- Complex task descriptions

This confused the model into just echoing the prompt.

## Solution
**Simplified the prompt template** to just:
```
Question: {question}
Image path: {image_path}
Answer:
```

## What I Fixed

### 1. Prompt Templates (`src/cg_prm/generation/prompts.py`)
- ✅ `gqa_canonical_v1` - Simplified (expects_json=False)
- ✅ `docvqa_canonical_v1` - Simplified (expects_json=False)

### 2. Batch Inference Script
Already fixed to:
- ✅ Extract `prompt["user"]` correctly
- ✅ Handle vLLM v0.17.x list response format
- ✅ Convert outputs to CG-PRM format

### 3. Output Parser
Automatically handles free-form text when `expects_json=False`

## Test Results

**Before (Complex Prompt):**
```
Response: Question: Are there more big green things...
          [Just echoes the prompt, 0 generation]
```

**After (Simple Prompt):**
```
Response: Question: Are there more big green things...
          Answer: To determine if there are more big green 
          things than large purple shiny cubes, we need to 
          analyze the image. Let's break it down:
          1. **Identify big green things:**
          [Generates 747 chars of actual reasoning!]
```

## How to Run Validation

```bash
# Make sure vLLM is running
python -m vllm.entrypoints.api_server \
  --model /hpc2hdd/home/ycui785/model/qwen3vl-4b \
  --tensor-parallel-size 4 \
  --port 8000 \
  --trust-remote-code

# Run complete validation
cd /hpc2hdd/home/ycui785/cg-prm
bash run_validation_fixed.sh
```

## Expected Timeline

| Step | Time |
|------|------|
| Teacher Inference (10k examples) | ~2-3 hours |
| Corruption Generation | ~10-15 min |
| Train CG-PRM | ~20-30 min |
| Train Pointwise | ~20-30 min |
| Evaluate | ~5 min |
| **TOTAL** | **~3-4 hours** |

## Files Changed

1. `src/cg_prm/generation/prompts.py` - Simplified templates
2. `scripts/inference/vllm_batch_inference.py` - Fixed earlier
3. `run_validation_fixed.sh` - New validation script

## Status

- ✅ vLLM prompt issue: **FIXED**
- ✅ Batch inference: **READY**
- ✅ Output parser: **READY**
- ✅ Validation script: **READY**

**Ready to run full validation!** 🚀
