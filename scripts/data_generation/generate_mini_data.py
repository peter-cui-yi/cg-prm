#!/usr/bin/env python3
"""Generate mini dataset for CG-PRM validation."""

import json
import random
from pathlib import Path

random.seed(42)

# Mini dataset sizes
CLEVR_CLEAN_TARGET = 1000
DOCVA_CLEAN_TARGET = 1500
F5_FRACTION = 0.5  # 50% of clean traces get F5 counterfactual

def generate_mock_trace(example_id, benchmark, answer, num_steps=3):
    """Generate a mock clean trace for testing."""
    steps = []
    step_types = ["locate", "read", "relate", "compute", "answer"]

    for i in range(num_steps):
        steps.append({
            "step_id": i + 1,
            "step_text": f"Step {i+1} reasoning for {example_id}",
            "step_type": step_types[min(i, len(step_types)-1)],
            "grounding_ref": f"region_{i}_{example_id}",
            "evidence_value": f"value_{i}",
            "label": 1,
            "error_type": "none"
        })

    return {
        "trace_id": f"trace_{example_id}",
        "example_id": example_id,
        "benchmark": benchmark,
        "image_path": f"/data/{benchmark}/{example_id}.jpg",
        "question": f"Question for {example_id}",
        "gold_answer": answer,
        "predicted_answer": answer,
        "steps": steps,
        "trace_mode": "canonical"
    }

def generate_f5_counterfactual(clean_trace):
    """Generate F5 counterfactual: correct answer, wrong evidence."""
    import copy
    cf = copy.deepcopy(clean_trace)
    cf["trace_id"] = f"{clean_trace['trace_id']}_f5"

    # Corrupt one step's grounding while keeping answer same
    corrupt_step_idx = random.randint(0, len(cf["steps"]) - 1)
    cf["steps"][corrupt_step_idx]["grounding_ref"] = f"wrong_region_{clean_trace['example_id']}"
    cf["steps"][corrupt_step_idx]["label"] = 0
    cf["steps"][corrupt_step_idx]["error_type"] = "wrong_intermediate_evidence"

    return cf

def construct_iso_answer_pair(clean_trace, counterfactual):
    """Construct pair with first divergence point."""
    # Find first step where labels differ
    t_star = None
    for i, (s1, s2) in enumerate(zip(clean_trace["steps"], counterfactual["steps"])):
        if s1["label"] != s2["label"]:
            t_star = i + 1  # step_id is 1-indexed
            break

    if t_star is None:
        t_star = 1  # Default to first step

    return {
        "positive": clean_trace,
        "negative": counterfactual,
        "t_star": t_star,
        "family": "f5_correct_answer_wrong_evidence"
    }

def main():
    output_dir = Path("data/mini")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating mini dataset in {output_dir}")

    # Generate CLEVR clean traces
    print(f"Generating {CLEVR_CLEAN_TARGET} CLEVR clean traces...")
    clevr_clean = []
    for i in range(CLEVR_CLEAN_TARGET):
        trace = generate_mock_trace(f"clevr_{i:04d}", "clevr", f"answer_{i % 10}", num_steps=3)
        clevr_clean.append(trace)

    # Generate DocVQA clean traces
    print(f"Generating {DOCVA_CLEAN_TARGET} DocVQA clean traces...")
    docvqa_clean = []
    for i in range(DOCVA_CLEAN_TARGET):
        trace = generate_mock_trace(f"docvqa_{i:04d}", "docvqa", f"answer_{i % 10}", num_steps=4)
        docvqa_clean.append(trace)

    # Save clean traces
    with open(output_dir / "clevr_clean.jsonl", "w") as f:
        for trace in clevr_clean:
            f.write(json.dumps(trace) + "\n")

    with open(output_dir / "docvqa_clean.jsonl", "w") as f:
        for trace in docvqa_clean:
            f.write(json.dumps(trace) + "\n")

    # Generate F5 counterfactuals
    print("Generating F5 counterfactuals...")
    clevr_cf = []
    docvqa_cf = []

    clevr_cf_count = int(CLEVR_CLEAN_TARGET * F5_FRACTION)
    docvqa_cf_count = int(DOCVA_CLEAN_TARGET * F5_FRACTION)

    for trace in clevr_clean[:clevr_cf_count]:
        clevr_cf.append(generate_f5_counterfactual(trace))

    for trace in docvqa_clean[:docvqa_cf_count]:
        docvqa_cf.append(generate_f5_counterfactual(trace))

    # Construct iso-answer pairs
    print("Constructing iso-answer pairs...")
    clevr_pairs = []
    docvqa_pairs = []

    for i, trace in enumerate(clevr_clean[:clevr_cf_count]):
        if i < len(clevr_cf):
            clevr_pairs.append(construct_iso_answer_pair(trace, clevr_cf[i]))

    for i, trace in enumerate(docvqa_clean[:docvqa_cf_count]):
        if i < len(docvqa_cf):
            docvqa_pairs.append(construct_iso_answer_pair(trace, docvqa_cf[i]))

    # Split into train/test (80/20)
    random.shuffle(clevr_pairs)
    random.shuffle(docvqa_pairs)

    clevr_split = int(len(clevr_pairs) * 0.8)
    docvqa_split = int(len(docvqa_pairs) * 0.8)

    train_pairs = clevr_pairs[:clevr_split] + docvqa_pairs[:docvqa_split]
    test_pairs = clevr_pairs[clevr_split:] + docvqa_pairs[docvqa_split:]

    # Save pairs
    with open(output_dir / "train_pairs.jsonl", "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")

    with open(output_dir / "test_pairs.jsonl", "w") as f:
        for pair in test_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n=== Mini Dataset Summary ===")
    print(f"CLEVR clean: {len(clevr_clean)}")
    print(f"DocVQA clean: {len(docvqa_clean)}")
    print(f"CLEVR F5 counterfactuals: {len(clevr_cf)}")
    print(f"DocVQA F5 counterfactuals: {len(docvqa_cf)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
