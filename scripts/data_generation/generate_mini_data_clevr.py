#!/usr/bin/env python3
"""Generate mini dataset for CG-PRM using REAL CLEVR images and scene metadata."""

import json
import random
import os
from pathlib import Path

random.seed(42)


def _find_clevr_scene_json_paths(clevr_root):
    """Return scene JSON paths for standard CLEVR layouts.

    Official zip has either a single top-level scenes.json, or
    scenes/CLEVR_train_scenes.json + scenes/CLEVR_val_scenes.json
    (and optionally CLEVR_test_scenes.json).
    """
    clevr_root = Path(clevr_root)
    single = clevr_root / "scenes.json"
    if single.is_file():
        return [single]
    scenes_dir = clevr_root / "scenes"
    if scenes_dir.is_dir():
        paths = sorted(scenes_dir.glob("CLEVR_*_scenes.json"))
        if paths:
            return paths
        paths = sorted(scenes_dir.glob("*_scenes.json"))
        if paths:
            return paths
    return []


def _resolve_clevr_dir():
    """Pick CLEVR root that contains scene JSON (handles .../CLEVR/CLEVR_v1.0)."""
    raw = os.environ.get("CLEVR_DIR", os.path.expanduser("~/datasets/CLEVR_v1.0"))
    raw = os.path.expanduser(raw)
    parent = os.path.dirname(raw.rstrip(os.sep))
    # Common layouts: DATA/CLEVR_v1.0, DATA/CLEVR_v1.0/CLEVR_v1.0, DATA/CLEVR/CLEVR_v1.0
    candidates = [
        raw,
        os.path.join(raw, "CLEVR_v1.0"),
        os.path.join(parent, "CLEVR", "CLEVR_v1.0"),
    ]
    for cand in candidates:
        if _find_clevr_scene_json_paths(cand):
            return cand
    return raw


# CLEVR paths
CLEVR_DIR = _resolve_clevr_dir()
CLEVR_IMAGES_DIR = os.path.join(CLEVR_DIR, "images")

# Dataset sizes (scale up for full experiment)
TRAIN_TARGET = 5000  # 5000 training pairs (was 500)
TEST_TARGET = 500    # 500 test pairs (was 100)

def load_clevr_scenes():
    """Load CLEVR scene metadata (ground truth objects, attributes, relations)."""
    paths = _find_clevr_scene_json_paths(CLEVR_DIR)
    if not paths:
        raise FileNotFoundError(
            "No CLEVR scene JSON found. Expected "
            f"{CLEVR_DIR}/scenes.json or {CLEVR_DIR}/scenes/CLEVR_*_scenes.json"
        )

    scenes = {}
    for scene_path in paths:
        print(f"Loading CLEVR scenes from {scene_path}...")
        with open(scene_path, 'r') as f:
            data = json.load(f)

        # scenes.json is a dict with "scenes" key containing list of scene infos
        for scene_info in data.get("scenes", []):
            image_filename = scene_info.get("image_filename", "")
            if image_filename:
                scenes[image_filename] = scene_info

    print(f"Loaded {len(scenes)} CLEVR scenes")
    return scenes

def get_random_scene_pair(scenes_list):
    """Get two scenes with overlapping objects for contrastive pairing."""
    scene1 = random.choice(scenes_list)
    scene2 = random.choice(scenes_list)
    return scene1, scene2

def generate_reasoning_trace(scene_info, question_type="count"):
    """Generate a structured reasoning trace from CLEVR scene metadata."""
    image_filename = scene_info.get("image_filename", "")
    objects = scene_info.get("objects", [])

    if not objects:
        return None

    # Generate different trace types based on question type
    steps = []

    if question_type == "count":
        # Counting question: locate objects -> filter by attribute -> count
        target_attr = random.choice(["size", "color", "material", "shape"])
        target_value = random.choice(list(set(
            obj.get(target_attr) for obj in objects if obj.get(target_attr)
        )))

        # Find matching objects
        matching = [obj for obj in objects if obj.get(target_attr) == target_value]
        count = len(matching)

        # Step 1: Locate all objects
        steps.append({
            "step_id": 1,
            "step_text": f"Locate all objects in the scene",
            "step_type": "locate",
            "grounding_ref": json.dumps({"type": "all_objects", "count": len(objects)}),
            "evidence_value": f"{len(objects)} objects",
            "label": 1,
            "error_type": "none"
        })

        # Step 2: Filter by attribute
        steps.append({
            "step_id": 2,
            "step_text": f"Filter objects by {target_attr}={target_value}",
            "step_type": "read",
            "grounding_ref": json.dumps({"type": "attribute_filter", "attr": target_attr, "value": target_value}),
            "evidence_value": target_value,
            "label": 1,
            "error_type": "none"
        })

        # Step 3: Count
        steps.append({
            "step_id": 3,
            "step_text": f"Count the filtered objects",
            "step_type": "compute",
            "grounding_ref": json.dumps({"type": "count", "result": count}),
            "evidence_value": str(count),
            "label": 1,
            "error_type": "none"
        })

        # Step 4: Answer
        steps.append({
            "step_id": 4,
            "step_text": f"The answer is {count}",
            "step_type": "answer",
            "grounding_ref": json.dumps({"type": "final_answer", "value": count}),
            "evidence_value": str(count),
            "label": 1,
            "error_type": "none"
        })

        question = f"How many objects are {target_value}?"
        answer = str(count)

    elif question_type == "exist":
        # Existence question
        target_obj = random.choice(objects)
        shape = target_obj.get("shape", "cube")
        color = target_obj.get("color", "red")

        steps.append({
            "step_id": 1,
            "step_text": f"Search for a {color} {shape} in the scene",
            "step_type": "locate",
            "grounding_ref": json.dumps({"type": "object_search", "shape": shape, "color": color}),
            "evidence_value": f"{color} {shape}",
            "label": 1,
            "error_type": "none"
        })

        exists = any(o.get("shape") == shape and o.get("color") == color for o in objects)

        steps.append({
            "step_id": 2,
            "step_text": f"{'Found' if exists else 'Did not find'} a {color} {shape}",
            "step_type": "read",
            "grounding_ref": json.dumps({"type": "existence_check", "result": exists}),
            "evidence_value": str(exists),
            "label": 1,
            "error_type": "none"
        })

        steps.append({
            "step_id": 3,
            "step_text": f"The answer is {'yes' if exists else 'no'}",
            "step_type": "answer",
            "grounding_ref": json.dumps({"type": "final_answer", "value": exists}),
            "evidence_value": "yes" if exists else "no",
            "label": 1,
            "error_type": "none"
        })

        question = f"Is there a {color} {shape}?"
        answer = "yes" if exists else "no"

    elif question_type == "relation":
        # Relation question (spatial)
        if len(objects) < 2:
            return None

        obj1 = random.choice(objects)
        obj2 = random.choice([o for o in objects if o != obj1])

        # Check actual relation
        obj1_pos = obj1.get("pixel_coords", [50, 50, 50])[:2]
        obj2_pos = obj2.get("pixel_coords", [50, 50, 50])[:2]

        # Simple spatial relation based on position
        if obj1_pos[0] < obj2_pos[0]:
            relation = "left of"
        else:
            relation = "right of"

        steps.append({
            "step_id": 1,
            "step_text": f"Locate the {obj1.get('color', 'red')} {obj1.get('shape', 'cube')}",
            "step_type": "locate",
            "grounding_ref": json.dumps({"type": "object", "id": 0, **obj1}),
            "evidence_value": f"{obj1.get('color')} {obj1.get('shape')}",
            "label": 1,
            "error_type": "none"
        })

        steps.append({
            "step_id": 2,
            "step_text": f"Locate the {obj2.get('color', 'blue')} {obj2.get('shape', 'sphere')}",
            "step_type": "locate",
            "grounding_ref": json.dumps({"type": "object", "id": 1, **obj2}),
            "evidence_value": f"{obj2.get('color')} {obj2.get('shape')}",
            "label": 1,
            "error_type": "none"
        })

        steps.append({
            "step_id": 3,
            "step_text": f"Determine spatial relation: object 1 is {relation} object 2",
            "step_type": "relate",
            "grounding_ref": json.dumps({"type": "spatial_relation", "relation": relation}),
            "evidence_value": relation,
            "label": 1,
            "error_type": "none"
        })

        steps.append({
            "step_id": 4,
            "step_text": f"The answer is {relation}",
            "step_type": "answer",
            "grounding_ref": json.dumps({"type": "final_answer", "value": relation}),
            "evidence_value": relation,
            "label": 1,
            "error_type": "none"
        })

        question = f"What is the spatial relation between the {obj1.get('color', 'red')} {obj1.get('shape', 'cube')} and the {obj2.get('color', 'blue')} {obj2.get('shape', 'sphere')}?"
        answer = relation

    else:
        return None

    # Get image path (CLEVR_train_*.png / CLEVR_val_*.png / CLEVR_test_*.png)
    fn = image_filename.lower()
    if "train" in fn:
        split = "train"
    elif "test" in fn:
        split = "test"
    else:
        split = "val"
    image_path = os.path.join(CLEVR_IMAGES_DIR, split, image_filename)

    return {
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "steps": steps,
        "question_type": question_type,
        "scene_info": scene_info
    }

def generate_f5_counterfactual(trace):
    """Generate F5 counterfactual: correct answer, wrong intermediate evidence."""
    import copy
    cf = copy.deepcopy(trace)

    # Corrupt step 2 (filter/compute step) while keeping answer same
    if len(cf["steps"]) >= 2:
        corrupt_step = cf["steps"][1]
        original_ref = corrupt_step["grounding_ref"]
        original_value = corrupt_step["evidence_value"]

        # Change the grounding_ref to wrong attribute/value
        corrupt_step["grounding_ref"] = f"wrong_{original_ref}"
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
        "example_id": trace["example_id"],
        "benchmark": trace["benchmark"],
        "image_path": trace["image_path"],
        "question": trace["question"],
        "gold_answer": trace["gold_answer"],
        "predicted_answer": trace["predicted_answer"],
        "steps": trace["steps"],
        "trace_mode": "canonical"
    }
    return cleaned

def main():
    output_dir = Path("data/mini_clevr")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("CLEVR Mini Dataset Generator")
    print("=" * 50)
    scene_paths = _find_clevr_scene_json_paths(CLEVR_DIR)
    print(f"CLEVR directory: {CLEVR_DIR}")
    print(f"Images directory: {CLEVR_IMAGES_DIR}")
    print(f"Scenes file(s): {', '.join(str(p) for p in scene_paths) or '(none found)'}")
    print("")

    # Check CLEVR exists
    if not scene_paths:
        print(f"ERROR: No CLEVR scene JSON found under {CLEVR_DIR}")
        print("Expected scenes.json or scenes/CLEVR_*_scenes.json (official CLEVR zip layout).")
        print("Run: bash scripts/download_clevr.sh ~/datasets")
        print("  or set CLEVR_DIR to your v1.0 root, e.g. ~/datasets/CLEVR/CLEVR_v1.0")
        return

    # Load scenes
    scenes_dict = load_clevr_scenes()
    scenes_list = list(scenes_dict.values())

    if not scenes_list:
        print("ERROR: No scenes loaded")
        return

    # Generate traces
    print(f"\nGenerating {TRAIN_TARGET} training traces...")
    clean_traces = []

    question_types = ["count", "exist", "relation"]

    for i in range(TRAIN_TARGET):
        qtype = random.choice(question_types)
        trace = generate_reasoning_trace(random.choice(scenes_list), qtype)
        if trace:
            trace["trace_id"] = f"clevr_train_{i:04d}"
            trace["example_id"] = f"clevr_{i:04d}"
            trace["benchmark"] = "clevr"
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

    # Clean and save
    print(f"Saving to {output_dir}...")

    # Save clean traces
    with open(output_dir / "clean_traces.jsonl", "w") as f:
        for trace in clean_traces:
            cleaned = clean_trace_for_training(trace)
            f.write(json.dumps(cleaned) + "\n")

    # Save train pairs
    with open(output_dir / "train_pairs.jsonl", "w") as f:
        for pair in train_pairs:
            cleaned_pair = {
                "positive": clean_trace_for_training(pair["positive"]),
                "negative": clean_trace_for_training(pair["negative"]),
                "t_star": pair["t_star"],
                "family": pair["family"]
            }
            f.write(json.dumps(cleaned_pair) + "\n")

    # Save test pairs
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
    print("CLEVR Mini Dataset Generated!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"  Clean traces: {len(clean_traces)}")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    print("")
    print("Verify image paths exist:")
    sample = clean_traces[0]
    print(f"  Sample image: {sample['image_path']}")
    print(f"  Exists: {os.path.exists(sample['image_path'])}")
    print("")

if __name__ == "__main__":
    main()
