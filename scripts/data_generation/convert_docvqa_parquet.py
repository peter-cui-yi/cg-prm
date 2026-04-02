#!/usr/bin/env python3
"""Convert HuggingFace DocVQA parquet files to JSON format."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def convert_docvqa_parquet_to_json(
    parquet_dir: Path,
    output_json_path: Path,
    output_ocr_path: Path,
):
    """Convert DocVQA parquet files to JSON format."""
    
    print(f"Loading DocVQA from: {parquet_dir}")
    
    # Find all parquet files
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load and concatenate all parquet files
    dfs = []
    for pf in tqdm(parquet_files, desc="Loading parquet files"):
        df = pd.read_parquet(pf)
        dfs.append(df)
    
    dataset = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dataset)} examples")
    
    # Convert to DocVQA JSON format
    annotations = []
    questions = []
    ocr_data = []
    
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Converting"):
        # Extract question
        question_id = str(row.get("questionId", i))
        question_text = str(row.get("question", ""))
        # Use docId as the image field (pipeline expects "image" field)
        image_name = str(row.get("docId", f"doc_{i}"))
        
        question_obj = {
            "questionId": question_id,
            "question": question_text,
            "image": image_name,  # Pipeline expects "image" field
        }
        questions.append(question_obj)
        
        # Extract answer/annotation - handle numpy arrays properly
        answers_raw = row.get("answers", [])
        # Convert numpy array or other formats to list
        if hasattr(answers_raw, 'tolist'):
            answers = answers_raw.tolist()
        elif isinstance(answers_raw, str):
            answers = [answers_raw]
        elif isinstance(answers_raw, (list, tuple)):
            answers = list(answers_raw)
        else:
            answers = []
        
        # Clean up answers
        answers = [str(a).strip() for a in answers if a is not None and str(a).strip()]
        
        answer = answers[0] if answers else ""
        annotation_obj = {
            "questionId": question_id,
            "image": image_name,  # Use same field name
            "answer": answer,
            "answers": [{"answer": a} for a in answers] if answers else [],
        }
        annotations.append(annotation_obj)
        
        # Extract OCR if available (not in this dataset)
        # OCR data would need to be extracted separately
    
    # Save questions
    print(f"Saving {len(questions)} questions to {output_json_path}")
    questions_data = {
        "data": questions,
        "annotations": {
            "questions": questions,
            "annotations": annotations,
        }
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(questions_data, f, indent=2)
    
    # Save OCR data
    if ocr_data:
        print(f"Saving {len(ocr_data)} OCR records to {output_ocr_path}")
        with open(output_ocr_path, "w") as f:
            json.dump({"data": ocr_data}, f, indent=2)
    else:
        print("No OCR data found")
    
    print("Conversion complete!")
    print(f"  Questions: {len(questions)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  OCR records: {len(ocr_data)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert DocVQA parquet to JSON")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/hpc2hdd/home/ycui785/datasets/DocVQA/DocVQA"),
        help="Input directory with parquet files",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/hpc2hdd/home/ycui785/datasets/DocVQA/train_v1.0.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--output-ocr",
        type=Path,
        default=Path("/hpc2hdd/home/ycui785/datasets/DocVQA/train_v1.0.ocr.json"),
        help="Output OCR JSON file",
    )
    
    args = parser.parse_args()
    convert_docvqa_parquet_to_json(args.input_dir, args.output_json, args.output_ocr)
