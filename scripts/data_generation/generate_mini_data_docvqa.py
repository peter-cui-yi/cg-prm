#!/usr/bin/env python3
"""Generate mini dataset for CG-PRM using DocVQA (HuggingFace Parquet format)."""

import json
import random
import os
from pathlib import Path

random.seed(42)

# DocVQA paths
DOCVQA_DIR = os.environ.get("DOCVQA_DIR", os.path.expanduser("~/datasets/DocVQA"))
DOCVQA_IMAGES_DIR = os.path.join(DOCVQA_DIR, "documents")

# Dataset sizes (scale up for full experiment)
TRAIN_TARGET = 3000  # 3000 training pairs (was 300)
TEST_TARGET = 300    # 300 test pairs (was 50)

def load_docvqa_from_parquet():
    """Load DocVQA from HuggingFace parquet format using datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    parquet_dir = Path(DOCVQA_DIR) / "DocVQA"
    train_parquets = sorted(parquet_dir.glob("train-*.parquet"))

    if train_parquets:
        print(f"Loading DocVQA from local parquet ({len(train_parquets)} train shards)...")
        ds = load_dataset(
            "parquet",
            data_files={"train": [str(p) for p in train_parquets]},
            split="train",
        )
    else:
        print("Loading DocVQA from HuggingFace (parquet)...")
        print("This may take a few minutes on first load...")
        # Hub config 'DocVQA' only registers validation + test splits (train is local shards only).
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

    print(f"Loaded {len(ds)} samples")

    # Build metadata without decoding images (faster, lower memory)
    meta_ds = ds.remove_columns(["image"]) if "image" in ds.column_names else ds

    # Lightweight rows with dataset index only (avoid holding ~40k PIL images in memory)
    qa_list = []
    for idx in range(len(meta_ds)):
        item = meta_ds[idx]
        qid = str(item.get("question_id") or item.get("questionId") or "")
        doc_id = item.get("docId", "")
        image_id = item.get("image_id") or (f"{doc_id}_{qid}.png" if qid else "")
        qa_list.append({
            "_idx": idx,
            "question_id": qid,
            "image_id": image_id,
            "question": item.get("question", ""),
            "answers": item.get("answers", [item.get("answer", "")]),
        })

    # Build image_id to QA mapping
    images = {}
    for qa in qa_list:
        img_id = qa.get("image_id", "")
        if img_id not in images:
            images[img_id] = []
        images[img_id].append(qa)

    print(f"Found {len(images)} unique images")
    return images, qa_list, ds

def generate_reasoning_trace(qa_item, ds=None):
    """Generate a structured reasoning trace from DocVQA."""
    question = qa_item.get("question", "")
    answers = qa_item.get("answers", [])

    if isinstance(answers, str):
        answers = [answers]

    if not answers or answers == [""]:
        return None

    # Most common answer
    answer = max(set(answers), key=answers.count) if answers else "unknown"

    # Get image info
    image_id = qa_item.get("image_id", "")

    # Generate multi-step trace for document reading
    steps = []

    # Step 1: Locate document region
    steps.append({
        "step_id": 1,
        "step_text": f"Locate the document in the image",
        "step_type": "locate",
        "grounding_ref": json.dumps({"type": "document_region", "image_id": image_id}),
        "evidence_value": "document detected",
        "label": 1,
        "error_type": "none"
    })

    # Step 2: OCR / Read text
    steps.append({
        "step_id": 2,
        "step_text": f"Read the document text to answer: {question[:50]}",
        "step_type": "read",
        "grounding_ref": json.dumps({"type": "ocr_text", "question": question[:50]}),
        "evidence_value": question[:50],
        "label": 1,
        "error_type": "none"
    })

    # Step 3: Extract answer from OCR
    steps.append({
        "step_id": 3,
        "step_text": f"Extract the answer from the document text",
        "step_type": "extract",
        "grounding_ref": json.dumps({"type": "answer_extraction", "answer": answer}),
        "evidence_value": str(answer),
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

    # Image path - parquet rows embed PIL images; write to documents/ on first use
    image_path = os.path.join(DOCVQA_IMAGES_DIR, image_id)
    pil = qa_item.get("_pil")
    if pil is None and ds is not None and qa_item.get("_idx") is not None:
        pil = ds[qa_item["_idx"]].get("image")
    if pil is not None and image_id and not os.path.exists(image_path):
        os.makedirs(DOCVQA_IMAGES_DIR, exist_ok=True)
        if getattr(pil, "mode", None) in ("RGBA", "P"):
            pil = pil.convert("RGB")
        pil.save(image_path)

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
        "benchmark": "docvqa",
        "image_path": trace["image_path"],
        "question": trace["question"],
        "gold_answer": trace["gold_answer"],
        "predicted_answer": trace["predicted_answer"],
        "steps": trace["steps"],
        "trace_mode": "canonical"
    }
    return cleaned

def main():
    output_dir = Path("data/mini_docvqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("DocVQA Mini Dataset Generator (HuggingFace Parquet)")
    print("=" * 50)
    print(f"DocVQA directory: {DOCVQA_DIR}")
    print("")

    # Check DocVQA exists
    if not os.path.exists(DOCVQA_DIR) or not os.listdir(DOCVQA_DIR):
        print(f"ERROR: DocVQA directory not found at {DOCVQA_DIR}")
        print("Run: bash scripts/download_docvqa_hf.sh ~/datasets")
        return

    # Load DocVQA from parquet
    try:
        images_dict, qa_list, ds = load_docvqa_from_parquet()
    except Exception as e:
        print(f"ERROR loading DocVQA: {e}")
        print("Make sure DocVQA is properly downloaded from HuggingFace")
        return

    # Generate traces
    print(f"\nGenerating {TRAIN_TARGET} training traces...")
    clean_traces = []

    for i in range(TRAIN_TARGET):
        qa = random.choice(qa_list)
        trace = generate_reasoning_trace(qa, ds=ds)
        if trace:
            trace["trace_id"] = f"docvqa_train_{i:04d}"
            trace["example_id"] = f"docvqa_{i:04d}"
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
    print("DocVQA Mini Dataset Generated!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"  Clean traces: {len(clean_traces)}")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    print("")

    # Verify some image paths
    if clean_traces:
        sample = clean_traces[0]
        print("Sample image verification:")
        print(f"  Path: {sample['image_path']}")
        print(f"  Exists: {os.path.exists(sample['image_path'])}")
    print("")

if __name__ == "__main__":
    main()
