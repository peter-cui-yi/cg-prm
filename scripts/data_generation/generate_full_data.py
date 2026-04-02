#!/usr/bin/env python3
"""Generate full-scale CLEVR + DocVQA datasets with all corruptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_prm.corruption import (
    generate_corrupted_traces,
    generate_cross_corruptor_traces,
    generate_wrong_use_traces,
)
from cg_prm.data.docvqa import write_docvqa_manifest
from cg_prm.data.gqa import write_gqa_manifest
from cg_prm.data.manifests import load_manifest
from cg_prm.data.schema import TraceRecord, read_jsonl, write_jsonl
from cg_prm.generation.teacher import (
    GenerationConfig,
    TeacherOutput,
    build_teacher_requests,
    parse_teacher_output,
)
from cg_prm.training.dataset_builder import (
    build_pairwise_dataset,
    build_pointwise_dataset,
    load_traces,
    write_pairwise_dataset,
    write_pointwise_dataset,
)
from cg_prm.verification.validators import annotate_trace_with_validation, validate_trace


def _resolve_path(value: str | None, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() and base_dir:
        path = base_dir / path
    return path


def prepare_teacher_requests(
    manifest_paths: dict[str, Path],
    output_dir: Path,
    model_name: str,
    prompt_id_prefix: str,
) -> dict[str, Path]:
    """Prepare teacher requests for all manifests."""
    
    print("\n=== Preparing Teacher Requests ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    request_paths = {}
    
    for name, manifest_path in manifest_paths.items():
        if "_val" in name:
            continue  # Skip validation sets for training
        
        print(f"Processing {name}...")
        manifest = load_manifest(manifest_path)
        
        benchmark = "gqa" if "clevr" in name else "docvqa"
        prompt_id = f"{benchmark}_canonical_v1"
        
        config = GenerationConfig(
            model_name=model_name,
            prompt_id=prompt_id,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            seed=0,
        )
        
        requests = build_teacher_requests(manifest, config)
        output_path = output_dir / f"{name}_requests.jsonl"
        write_jsonl(output_path, requests)
        
        request_paths[name] = output_path
        print(f"  {name}: {len(requests)} requests")
    
    return request_paths


def build_manifests(
    clevr_dir: Path,
    docvqa_dir: Path,
    output_dir: Path,
    clevr_limit: int | None = None,
    docvqa_limit: int | None = None,
) -> dict[str, Path]:
    """Build manifests for CLEVR and DocVQA."""
    
    print("\n=== Building Manifests ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CLEVR manifests
    print("Building CLEVR manifest...")
    clevr_train_manifest = write_gqa_manifest(
        output_path=output_dir / "clevr_train.jsonl",
        questions_path=str(clevr_dir / "CLEVR_v1.0/questions/CLEVR_train_questions.json"),
        image_root=str(clevr_dir / "CLEVR_v1.0/images/train"),
        scene_graphs_path=str(clevr_dir / "CLEVR_v1.0/scenes/CLEVR_train_scenes.json"),
        split="train",
    )
    if clevr_limit:
        clevr_train_manifest = clevr_train_manifest[:clevr_limit]
        write_jsonl(output_dir / "clevr_train.jsonl", clevr_train_manifest)
    
    clevr_val_manifest = write_gqa_manifest(
        output_path=output_dir / "clevr_val.jsonl",
        questions_path=str(clevr_dir / "CLEVR_v1.0/questions/CLEVR_val_questions.json"),
        image_root=str(clevr_dir / "CLEVR_v1.0/images/val"),
        scene_graphs_path=str(clevr_dir / "CLEVR_v1.0/scenes/CLEVR_val_scenes.json"),
        split="val",
    )
    
    print(f"  CLEVR train: {len(clevr_train_manifest)} examples")
    print(f"  CLEVR val: {len(clevr_val_manifest)} examples")
    
    # DocVQA manifests (optional)
    docvqa_train_manifest = []
    docvqa_val_manifest = []
    
    docvqa_questions_train = docvqa_dir / "train_v1.0.json"
    docvqa_questions_val = docvqa_dir / "val_v1.0.json"
    docvqa_ocr_train = docvqa_dir / "train_v1.0.ocr.json"
    docvqa_ocr_val = docvqa_dir / "val_v1.0.ocr.json"
    docvqa_images_train = docvqa_dir / "documents"
    docvqa_images_val = docvqa_dir / "documents"
    
    if docvqa_questions_train.exists():
        print("Building DocVQA manifest...")
        docvqa_train_manifest = write_docvqa_manifest(
            output_path=output_dir / "docvqa_train.jsonl",
            questions_path=str(docvqa_questions_train),
            image_root=str(docvqa_images_train),
            ocr_path=str(docvqa_ocr_train) if docvqa_ocr_train.exists() else None,
            split="train",
        )
        if docvqa_limit:
            docvqa_train_manifest = docvqa_train_manifest[:docvqa_limit]
            write_jsonl(output_dir / "docvqa_train.jsonl", docvqa_train_manifest)
        
        docvqa_val_manifest = []
        if docvqa_questions_val.exists():
            try:
                docvqa_val_manifest = write_docvqa_manifest(
                    output_path=output_dir / "docvqa_val.jsonl",
                    questions_path=str(docvqa_questions_val),
                    image_root=str(docvqa_images_val),
                    ocr_path=str(docvqa_ocr_val) if docvqa_ocr_val.exists() else None,
                    split="val",
                )
            except ValueError as e:
                print(f"  ⚠️  Skipping DocVQA val: {e}")
                print(f"     (test sets typically don't have ground truth answers)")
        
        print(f"  DocVQA train: {len(docvqa_train_manifest)} examples")
        print(f"  DocVQA val: {len(docvqa_val_manifest)} examples")
    else:
        print("  ⚠️  DocVQA JSON files not found - skipping DocVQA")
        print(f"     Expected: {docvqa_questions_train}")
        print(f"     Expected: {docvqa_ocr_train}")
        print("     Note: DocVQA requires manual download from https://docvqa.cs.cmu.edu/")
    
    return {
        "clevr_train": output_dir / "clevr_train.jsonl",
        "clevr_val": output_dir / "clevr_val.jsonl",
        "docvqa_train": output_dir / "docvqa_train.jsonl",
        "docvqa_val": output_dir / "docvqa_val.jsonl",
    }


def prepare_teacher_requests(
    manifest_paths: dict[str, Path],
    output_dir: Path,
    model_name: str,
    prompt_id_prefix: str,
) -> dict[str, Path]:
    """Prepare teacher requests for all manifests."""
    
    print("\n=== Preparing Teacher Requests ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    request_paths = {}
    
    for name, manifest_path in manifest_paths.items():
        if "_val" in name:
            continue  # Skip validation sets for training
        
        print(f"Processing {name}...")
        manifest = load_manifest(manifest_path)
        
        benchmark = "gqa" if "clevr" in name else "docvqa"
        prompt_id = f"{benchmark}_canonical_v1"
        
        config = GenerationConfig(
            model_name=model_name,
            prompt_id=prompt_id,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            seed=0,
        )
        
        requests = build_teacher_requests(manifest, config)
        output_path = output_dir / f"{name}_requests.jsonl"
        write_jsonl(output_path, requests)
        
        request_paths[name] = output_path
        print(f"  {name}: {len(requests)} requests")
    
    return request_paths


def parse_and_verify_traces(
    teacher_outputs_path: Path,
    manifest_path: Path,
    output_dir: Path,
    benchmark: str,
) -> tuple[list[TraceRecord], list[TraceRecord]]:
    """Parse teacher outputs and verify traces."""
    
    print(f"\n=== Parsing and Verifying {benchmark} Traces ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse outputs
    print("Parsing teacher outputs...")
    payloads = read_jsonl(teacher_outputs_path)
    traces = []
    failed = 0
    for payload in payloads:
        try:
            trace = parse_teacher_output(TeacherOutput.from_dict(payload))
            traces.append(trace)
        except Exception as e:
            failed += 1
            if failed <= 3:  # Show first 3 errors
                print(f"  ⚠️  Failed to parse {payload.get('example_id', 'unknown')}: {e}")
    
    if failed > 0:
        print(f"  Parsed {len(traces)} traces ({failed} failed)")
    else:
        print(f"  Parsed {len(traces)} traces")
    
    # Load manifest
    manifest = load_manifest(manifest_path)
    example_lookup = {example.example_id: example for example in manifest}
    
    # Verify traces
    print("Verifying traces...")
    verified = []
    rejected = []
    
    for trace in traces:
        example = example_lookup.get(trace.example_id)
        if example is None:
            continue
        
        validation = validate_trace(example, trace)
        annotated = annotate_trace_with_validation(trace, validation)
        
        if validation.passed:
            verified.append(annotated)
        else:
            rejected.append(annotated)
    
    # Save results
    verified_path = output_dir / f"{benchmark}_verified.jsonl"
    rejected_path = output_dir / f"{benchmark}_rejected.jsonl"
    write_jsonl(verified_path, verified)
    write_jsonl(rejected_path, rejected)
    
    print(f"  Verified: {len(verified)} ({100*len(verified)/len(traces):.1f}%)")
    print(f"  Rejected: {len(rejected)} ({100*len(rejected)/len(traces):.1f}%)")
    
    return verified, rejected


def generate_corruptions(
    verified_traces: list[TraceRecord],
    manifest_path: Path,
    output_dir: Path,
    benchmark: str,
    base_seed: int = 0,
) -> list[TraceRecord]:
    """Generate all corruption families for verified traces."""
    
    print(f"\n=== Generating Corruptions for {benchmark} ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest(manifest_path)
    example_lookup = {example.example_id: example for example in manifest}
    
    all_corrupted = []
    
    for index, trace in enumerate(verified_traces):
        example = example_lookup[trace.example_id]
        seed = base_seed + index
        
        # Generate main corruptions (F1-F7)
        main_corrupted = generate_corrupted_traces(example, trace, seed=seed)
        all_corrupted.extend(main_corrupted)
        
        # Generate cross-corruptor traces
        cross_corrupted = generate_cross_corruptor_traces(example, trace, seed=seed)
        all_corrupted.extend(cross_corrupted)
        
        # Generate wrong_use traces
        wrong_use = generate_wrong_use_traces(example, trace, seed=seed)
        all_corrupted.extend(wrong_use)
    
    # Save corrupted traces
    corrupted_path = output_dir / f"{benchmark}_corrupted.jsonl"
    write_jsonl(corrupted_path, all_corrupted)
    
    print(f"  Generated {len(all_corrupted)} corrupted traces")
    print(f"  Saved to: {corrupted_path}")
    
    return all_corrupted


def build_training_datasets(
    clean_traces: list[TraceRecord],
    corrupted_traces: list[TraceRecord],
    output_dir: Path,
    test_split: float = 0.1,
) -> dict[str, Path]:
    """Build pairwise and pointwise training datasets."""
    
    print("\n=== Building Training Datasets ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train/test
    import random
    random.seed(42)
    
    indices = list(range(len(clean_traces)))
    random.shuffle(indices)
    
    test_size = int(len(indices) * test_split)
    test_indices = set(indices[:test_size])
    train_indices = indices[test_size:]
    
    clean_train = [t for i, t in enumerate(clean_traces) if i in train_indices]
    clean_test = [t for i, t in enumerate(clean_traces) if i in test_indices]
    
    corrupted_train = [t for t in corrupted_traces if t.example_id in {c.example_id for c in clean_train}]
    corrupted_test = [t for t in corrupted_traces if t.example_id in {c.example_id for c in clean_test}]
    
    # Build datasets
    print("Building pointwise dataset...")
    pointwise_train = build_pointwise_dataset(clean_train, corrupted_train)
    pointwise_test = build_pointwise_dataset(clean_test, corrupted_test)
    
    print("Building pairwise dataset...")
    pairwise_train = build_pairwise_dataset(clean_train, corrupted_train)
    pairwise_test = build_pairwise_dataset(clean_test, corrupted_test)
    
    # Save datasets
    datasets = {
        "pointwise_train": output_dir / "pointwise_train.jsonl",
        "pointwise_val": output_dir / "pointwise_val.jsonl",
        "pairwise_train": output_dir / "pairwise_train.jsonl",
        "pairwise_val": output_dir / "pairwise_val.jsonl",
    }
    
    write_pointwise_dataset(datasets["pointwise_train"], pointwise_train)
    write_pointwise_dataset(datasets["pointwise_val"], pointwise_test)
    write_pairwise_dataset(datasets["pairwise_train"], pairwise_train)
    write_pairwise_dataset(datasets["pairwise_val"], pairwise_test)
    
    print(f"\nDataset statistics:")
    print(f"  Pointwise train: {len(pointwise_train)} pairs")
    print(f"  Pointwise val: {len(pointwise_test)} pairs")
    print(f"  Pairwise train: {len(pairwise_train)} pairs")
    print(f"  Pairwise val: {len(pairwise_test)} pairs")
    
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full-scale CG-PRM dataset")
    parser.add_argument("--clevr-dir", required=False, help="CLEVR dataset directory")
    parser.add_argument("--docvqa-dir", required=False, help="DocVQA dataset directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--clevr-limit", type=int, help="Limit CLEVR examples")
    parser.add_argument("--docvqa-limit", type=int, help="Limit DocVQA examples")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3VL-32B-Thinking")
    parser.add_argument("--vllm-server", default="http://localhost:8000")
    parser.add_argument("--skip-manifests", action="store_true")
    parser.add_argument("--skip-teacher-inference", action="store_true")
    parser.add_argument("--skip-corruptions", action="store_true")
    parser.add_argument("--benchmark", choices=["clevr", "docvqa", "both"], default="both")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CG-PRM Full-Scale Dataset Generation")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Benchmark: {args.benchmark}")
    print("")
    
    # Step 1: Build manifests (if not skipping)
    if not args.skip_manifests:
        if not args.clevr_dir or not args.docvqa_dir:
            print("ERROR: --clevr-dir and --docvqa-dir required when not skipping manifests")
            return 1
        manifest_paths = build_manifests(
            clevr_dir=Path(args.clevr_dir),
            docvqa_dir=Path(args.docvqa_dir),
            output_dir=output_dir / "manifests",
            clevr_limit=args.clevr_limit,
            docvqa_limit=args.docvqa_limit,
        )
    else:
        manifest_paths = {
            "clevr_train": output_dir / "manifests/clevr_train.jsonl",
            "clevr_val": output_dir / "manifests/clevr_val.jsonl",
            "docvqa_train": output_dir / "manifests/docvqa_train.jsonl",
            "docvqa_val": output_dir / "manifests/docvqa_val.jsonl",
        }
    
    # Step 2: Prepare teacher requests
    if not args.skip_teacher_inference:
        request_paths = prepare_teacher_requests(
            manifest_paths=manifest_paths,
            output_dir=output_dir / "teacher_requests",
            model_name=args.teacher_model,
            prompt_id_prefix="canonical",
        )
        
        # Step 3: Run teacher inference (external - just prepare commands)
        print("\n=== Teacher Inference Instructions ===")
        print("Launch vLLM server:")
        print(f"  bash scripts/launch_vllm_server.sh")
        print("\nRun batch inference:")
        for name, req_path in request_paths.items():
            if "_val" not in name:
                output_path = output_dir / "teacher_outputs" / f"{name}_outputs.jsonl"
                print(f"  python scripts/vllm_batch_inference.py \\")
                print(f"    --requests {req_path} \\")
                print(f"    --output {output_path} \\")
                print(f"    --server-url {args.vllm_server} \\")
                print(f"    --mode infer")
                print("")
        
        print("\nAfter inference completes, run this script again with:")
        print("  --skip-manifests --skip-teacher-inference")
        print("")
        return 0
    
    # Step 4: Parse and verify traces
    all_clean_traces = []
    all_corrupted_traces = []
    
    benchmarks = ["clevr", "docvqa"] if args.benchmark == "both" else [args.benchmark]
    
    for benchmark in benchmarks:
        manifest_path = manifest_paths[f"{benchmark}_train"]
        teacher_outputs_path = output_dir / "teacher_outputs" / f"{benchmark}_train_outputs.jsonl"
        
        if not teacher_outputs_path.exists():
            print(f"WARNING: Teacher outputs not found for {benchmark}")
            print(f"  Expected: {teacher_outputs_path}")
            continue
        
        verified, rejected = parse_and_verify_traces(
            teacher_outputs_path=teacher_outputs_path,
            manifest_path=manifest_path,
            output_dir=output_dir / "clean_traces",
            benchmark=benchmark,
        )
        
        all_clean_traces.extend(verified)
        
        # Step 5: Generate corruptions
        if not args.skip_corruptions:
            corrupted = generate_corruptions(
                verified_traces=verified,
                manifest_path=manifest_path,
                output_dir=output_dir / "corrupted_traces",
                benchmark=benchmark,
                base_seed=0,
            )
            all_corrupted_traces.extend(corrupted)
    
    # Step 6: Build training datasets
    if all_clean_traces and all_corrupted_traces:
        dataset_paths = build_training_datasets(
            clean_traces=all_clean_traces,
            corrupted_traces=all_corrupted_traces,
            output_dir=output_dir / "training_pairs",
            test_split=0.1,
        )
        
        # Summary
        print("\n" + "=" * 60)
        print("Full-Scale Dataset Generation Complete!")
        print("=" * 60)
        print(f"\nOutput directory: {output_dir}")
        print(f"\nFinal datasets:")
        for name, path in dataset_paths.items():
            count = sum(1 for _ in open(path))
            print(f"  {name}: {count} pairs")
        print(f"\nNext step: Train models")
        print(f"  python scripts/train_lora.py --config configs/full_cg_prm.json")
        print(f"  python scripts/train_lora.py --config configs/full_pointwise.json")
    else:
        print("\nWARNING: Insufficient data generated")
        print(f"  Clean traces: {len(all_clean_traces)}")
        print(f"  Corrupted traces: {len(all_corrupted_traces)}")
        print("\nPlease ensure teacher inference completed successfully")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
