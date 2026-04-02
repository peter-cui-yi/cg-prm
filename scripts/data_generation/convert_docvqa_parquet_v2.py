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
    
    # Find only train parquet files (exclude test files which don't have answers)
    parquet_files = sorted([f for f in parquet_dir.glob("train*.parquet")])
    if not parquet_files:
        # Fallback to all parquet files if no train files found
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
    questions = []
    annotations = []
    
    for idx in tqdm(range(len(dataset)), desc="Converting"):
        row = dataset.iloc[idx]
        
        # Extract question
        question_id = str(row["questionId"])
        question_text = str(row["question"])
        image_name = str(row["docId"])
        
        # Extract answer/annotation - handle numpy arrays
        answers_raw = row["answers"]
        if hasattr(answers_raw, 'tolist'):
            answers_list = answers_raw.tolist()
        elif isinstance(answers_raw, (list, tuple)):
            answers_list = list(answers_raw)
        elif isinstance(answers_raw, str):
            answers_list = [answers_raw]
        else:
            answers_list = []
        
        # Clean up answers
        answers = [str(a).strip() for a in answers_list if a is not None and str(a).strip()]
        
        # Create question object WITH answers embedded (pipeline expects this format)
        question_obj = {
            "questionId": question_id,
            "question": question_text,
            "image": image_name,
            "answers": answers if answers else [],  # Embed answers directly
        }
        questions.append(question_obj)
    
    # Save questions - simple format with embedded answers
    print(f"Saving {len(questions)} questions to {output_json_path}")
    questions_data = {
        "data": questions,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(questions_data, f, indent=2)
    
    # Save OCR data if we had any
    print("No OCR data found")
    
    print("Conversion complete!")
    print(f"  Questions: {len(questions)}")
    if questions:
        sample = questions[0]
        print(f"  Sample question: {sample['questionId']}")
        print(f"  Sample answer: {sample.get('answers', ['EMPTY'])[0] if sample.get('answers') else 'EMPTY'}")


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
