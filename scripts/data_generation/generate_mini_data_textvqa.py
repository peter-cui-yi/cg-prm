#!/usr/bin/env python3
"""Generate mini dataset for CG-PRM using TextVQA (DocVQA alternative).

TextVQA is similar to DocVQA but doesn't require registration.
It has images with text in natural scenes (signs, documents, products).
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

# TextVQA paths
TEXTVQA_DIR = os.environ.get("TEXTVQA_DIR", os.path.expanduser("~/datasets/TextVQA"))
TEXTVQA_JSON = os.path.join(TEXTVQA_DIR, "TextVQA_0.5.1_train.json")

# Mini dataset sizes
TRAIN_TARGET = 300  # 300 training pairs
TEST_TARGET = 50    # 50 test pairs

def load_textvqa_data():
    """Load TextVQA question-answer pairs."""
    if not os.path.exists(TEXTVQA_JSON):
        raise FileNotFoundError(f"TextVQA data not found at {TEXTVQA_JSON}")

    print(f"Loading TextVQA from {TEXTVQA_JSON}...")
    with open(TEXTVQA_JSON, 'r') as f:
        data = json.load(f)

    # Data structure: {"data": [{"question_id", "image_id", "question", "answers", ...}]}
    qa_list = data.get("data", [])
    print(f"Loaded {len(qa_list)} QA pairs")

    # Build image_id to QA mapping
    images = {}
    for qa in qa_list:
        img_id = qa.get("image_id", "")
        if img_id not in images:
            images[img_id] = []
        images[img_id].append(qa)

    print(f"Found {len(images)} unique images")
    return images, qa_list

def generate_reasoning_trace(qa_item, image_info):
    """Generate a structured reasoning trace from TextVQA."""
    question = qa_item.get("question", "")
    answers = qa_item.get("answers", [])
    image_id = qa_item.get("image_id", "")

    if not answers:
        return None

    # Most common answer
    answer = max(set(answers), key=answers.count)

    # Generate multi-step trace for text reading
    steps = []

    # Step 1: Locate text region
    steps.append({
        "step_id": 1,
        "step_text": f"Locate text regions in the image",
        "step_type": "locate",
        "grounding_ref": json.dumps({"type": "text_region", "image_id": image_id}),
        "evidence_value": "text detected",
        "label": 1,
        "error_type": "none"
    })

    # Step 2: Read text content
    steps.append({
        "step_id": 2,
        "step_text": f"Read the text content: {question[:50]}",
        "step_type": "read",
        "grounding_ref": json.dumps({"type": "ocr_text", "question": question[:50]}),
        "evidence_value": question[:50],
        "label": 1,
        "error_type": "none"
    })

    # Step 3: Extract answer from text
    steps.append({
        "step_id": 3,
        "step_text": f"Extract the answer from the text",
        "step_type": "extract",
        "grounding_ref": json.dumps({"type": "answer_extraction", "answer": answer}),
        "evidence_value": answer,
        "label": 1,
        "error_type": "none"
    })

    # Step 4: Final answer
    steps.append({
        "step_id": 4,
        "step_text": f"The answer is {answer}",
        "step_type": "answer",
        "grounding_ref": json.dumps({"type": "final_answer", "value": answer}),
        "evidence_value": str(answer),
        "label": 1,
        "error_type": "none"
    })

    # Image path (TextVQA uses Visual Genome images)
    # Image URL format: https://dl.fbaipublicfiles.com/textvqa/images/{image_id}.jpg
    # For local use, we'll use a placeholder that points to VG
    image_path = os.path.join(TEXTVQA_DIR, "train_val_images", f"{image_id}.jpg")

    return {
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "steps": steps,
        "question_id": qa_item.get("question_id", ""),
        "image_id": image_id
    }

def generate_f5_counterfactual(trace):
    """Generate F5 counterfactual: correct answer, wrong intermediate evidence."""
    import copy
    cf = copy.deepcopy(trace)

    # Corrupt step 2 (read step) while keeping answer same
    if len(cf["steps"]) >= 2:
        corrupt_step = cf["steps"][1]
        original_ref = corrupt_step["grounding_ref"]
        original_value = corrupt_step["evidence_value"]

        # Change the grounding_ref to wrong text
        corrupt_step["grounding_ref"] = f"wrong_text_{original_ref}"
        corrupt_step["evidence_value"] = f"wrong_{original_value}"
        corrupt_step["label"] = 0
        corrupt_step["error_type"] = "wrong_intermediate_evidence"

    cf["trace_id"] = f"{trace['trace_id']}_f5"
    return cf

def construct_pair(clean_trace, counterfactual):
    """Construct iso-answer pair with first divergence point."""
    t_star = None
    for i, (s1, s2) in enumerate(zip(clean_trace["steps"], counterfactual["steps"])):
        if s1["label"] != s2["label"]:
            t_star = i + 1
            break

    if t_star is None:
        t_star = 1

    return {
        "positive": clean_trace,
        "negative": counterfactual,
        "t_star": t_star,
        "family": "f5_correct_answer_wrong_evidence"
    }

def clean_trace_for_training(trace):
    """Remove non-serializable fields from trace."""
    cleaned = {
        "trace_id": trace["trace_id"],
        "example_id": trace.get("example_id", trace["trace_id"]),
        "benchmark": "textvqa",
        "image_path": trace["image_path"],
        "question": trace["question"],
        "gold_answer": trace["gold_answer"],
        "predicted_answer": trace["predicted_answer"],
        "steps": trace["steps"],
        "trace_mode": "canonical"
    }
    return cleaned

def main():
    output_dir = Path("data/mini_textvqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("TextVQA Mini Dataset Generator (DocVQA Alternative)")
    print("=" * 50)
    print(f"TextVQA directory: {TEXTVQA_DIR}")
    print(f"TextVQA JSON: {TEXTVQA_JSON}")
    print("")

    # Check TextVQA exists
    if not os.path.exists(TEXTVQA_JSON):
        print(f"ERROR: TextVQA data not found at {TEXTVQA_JSON}")
        print("Run: bash scripts/download_docvqa.sh ~/datasets")
        print("Or use CLEVR only: python scripts/generate_mini_data_clevr.py")
        return

    # Load TextVQA
    images_dict, qa_list = load_textvqa_data()

    # Generate traces
    print(f"\nGenerating {TRAIN_TARGET} training traces...")
    clean_traces = []

    for i in range(TRAIN_TARGET):
        qa = random.choice(qa_list)
        trace = generate_reasoning_trace(qa, images_dict.get(qa.get("image_id"), {}))
        if trace:
            trace["trace_id"] = f"textvqa_train_{i:04d}"
            trace["example_id"] = f"textvqa_{i:04d}"
            trace["gold_answer"] = trace["answer"]
            trace["predicted_answer"] = trace["answer"]
            clean_traces.append(trace)

    print(f"Generated {len(clean_traces)} valid traces")

    # Generate F5 counterfactuals
    print(f"Generating F5 counterfactuals...")
    cf_count = int(len(clean_traces) * 0.5)
    counterfactuals = []
    for i, trace in enumerate(clean_traces[:cf_count]):
        cf = generate_f5_counterfactual(trace)
        counterfactuals.append(cf)

    # Construct pairs
    print("Constructing iso-answer pairs...")
    pairs = []
    for i, trace in enumerate(clean_traces[:cf_count]):
        if i < len(counterfactuals):
            pair = construct_pair(trace, counterfactuals[i])
            pairs.append(pair)

    # Split train/test
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    # Save
    print(f"Saving to {output_dir}...")

    with open(output_dir / "clean_traces.jsonl", "w") as f:
        for trace in clean_traces:
            cleaned = clean_trace_for_training(trace)
            f.write(json.dumps(cleaned) + "\n")

    with open(output_dir / "train_pairs.jsonl", "w") as f:
        for pair in train_pairs:
            cleaned_pair = {
                "positive": clean_trace_for_training(pair["positive"]),
                "negative": clean_trace_for_training(pair["negative"]),
                "t_star": pair["t_star"],
                "family": pair["family"]
            }
            f.write(json.dumps(cleaned_pair) + "\n")

    with open(output_dir / "test_pairs.jsonl", "w") as f:
        for pair in test_pairs:
            cleaned_pair = {
                "positive": clean_trace_for_training(pair["positive"]),
                "negative": clean_trace_for_training(pair["negative"]),
                "t_star": pair["t_star"],
                "family": pair["family"]
            }
            f.write(json.dumps(cleaned_pair) + "\n")

    print("")
    print("=" * 50)
    print("TextVQA Mini Dataset Generated!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"  Clean traces: {len(clean_traces)}")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    print("")

if __name__ == "__main__":
    main()
