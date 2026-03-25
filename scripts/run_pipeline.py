#!/usr/bin/env python3
"""One-entry pipeline driver for CG-PRM data preparation and training-data build."""

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
from cg_prm.data.visualwebbench import write_visualwebbench_manifest
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


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_manifest(benchmark: str, payload: dict[str, Any]) -> tuple[list[Any], Path]:
    manifest_output = _resolve_path(payload["manifest_output"])
    assert manifest_output is not None
    _ensure_parent(manifest_output)

    if benchmark == "docvqa":
        manifest = write_docvqa_manifest(
            output_path=manifest_output,
            questions_path=payload["questions"],
            image_root=payload["images"],
            ocr_path=payload.get("ocr"),
            split=payload.get("split"),
        )
    elif benchmark == "gqa":
        manifest = write_gqa_manifest(
            output_path=manifest_output,
            questions_path=payload["questions"],
            image_root=payload["images"],
            scene_graphs_path=payload.get("scene_graphs"),
            split=payload.get("split"),
        )
    elif benchmark == "visualwebbench":
        manifest = write_visualwebbench_manifest(
            output_path=manifest_output,
            items_path=payload["items"],
            image_root=payload.get("images"),
            split=payload.get("split"),
        )
    else:
        raise ValueError(f"Unsupported benchmark `{benchmark}`.")
    return manifest, manifest_output


def _prepare_requests(
    manifest_path: Path,
    benchmark: str,
    benchmark_cfg: dict[str, Any],
    teacher_cfg: dict[str, Any],
) -> tuple[list[Any], Path]:
    manifest = load_manifest(manifest_path)
    if benchmark_cfg.get("request_limit") is not None:
        manifest = manifest[: int(benchmark_cfg["request_limit"])]
    config = GenerationConfig(
        model_name=str(teacher_cfg["model_name"]),
        prompt_id=str(benchmark_cfg["prompt_id"]),
        max_tokens=int(teacher_cfg.get("max_tokens", 1024)),
        temperature=float(teacher_cfg.get("temperature", 0.2)),
        top_p=float(teacher_cfg.get("top_p", 0.95)),
        seed=int(teacher_cfg.get("seed", 0)),
    )
    requests = build_teacher_requests(manifest, config)
    output_path = _resolve_path(benchmark_cfg["teacher_requests_output"])
    assert output_path is not None
    _ensure_parent(output_path)
    write_jsonl(output_path, requests)
    return requests, output_path


def _parse_teacher_outputs(teacher_outputs_path: Path, parsed_output_path: Path) -> list[TraceRecord]:
    payloads = read_jsonl(teacher_outputs_path)
    traces = [parse_teacher_output(TeacherOutput.from_dict(payload)) for payload in payloads]
    _ensure_parent(parsed_output_path)
    write_jsonl(parsed_output_path, traces)
    return traces


def _verify_traces(
    manifest_path: Path,
    traces: list[TraceRecord],
    *,
    verified_output_path: Path,
    rejected_output_path: Path,
) -> tuple[list[TraceRecord], list[TraceRecord]]:
    manifest = load_manifest(manifest_path)
    example_lookup = {example.example_id: example for example in manifest}
    verified: list[TraceRecord] = []
    rejected: list[TraceRecord] = []
    for trace in traces:
        example = example_lookup.get(trace.example_id)
        if example is None:
            raise KeyError(
                f"Trace `{trace.trace_id}` refers to unknown example `{trace.example_id}`."
            )
        validation = validate_trace(example, trace)
        annotated = annotate_trace_with_validation(trace, validation)
        if validation.passed:
            verified.append(annotated)
        else:
            rejected.append(annotated)
    _ensure_parent(verified_output_path)
    _ensure_parent(rejected_output_path)
    write_jsonl(verified_output_path, verified)
    write_jsonl(rejected_output_path, rejected)
    return verified, rejected


def _build_corruptions_for_benchmark(
    manifest_path: Path,
    verified_traces: list[TraceRecord],
    benchmark_cfg: dict[str, Any],
    *,
    base_seed: int,
) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    example_lookup = {example.example_id: example for example in manifest}

    main_traces: list[TraceRecord] = []
    cross_traces: list[TraceRecord] = []
    wrong_use_traces: list[TraceRecord] = []

    for index, trace in enumerate(verified_traces):
        example = example_lookup[trace.example_id]
        seed = base_seed + index
        main_traces.extend(generate_corrupted_traces(example, trace, seed=seed))
        cross_traces.extend(generate_cross_corruptor_traces(example, trace, seed=seed))
        wrong_use_traces.extend(generate_wrong_use_traces(example, trace, seed=seed))

    main_output = _resolve_path(benchmark_cfg["corrupted_main_output"])
    cross_output = _resolve_path(benchmark_cfg["corrupted_cross_output"])
    wrong_use_output = _resolve_path(benchmark_cfg["corrupted_wrong_use_output"])
    assert main_output is not None and cross_output is not None and wrong_use_output is not None
    for path in (main_output, cross_output, wrong_use_output):
        _ensure_parent(path)
    write_jsonl(main_output, main_traces)
    write_jsonl(cross_output, cross_traces)
    write_jsonl(wrong_use_output, wrong_use_traces)

    return {
        "main": len(main_traces),
        "cross": len(cross_traces),
        "wrong_use": len(wrong_use_traces),
    }


def _maybe_build_training_dataset(
    config: dict[str, Any],
    benchmark_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    training_cfg = dict(config.get("training_dataset") or {})
    if not training_cfg.get("enabled", True):
        return {"status": "disabled"}

    clean_paths: list[Path] = []
    corrupted_paths: list[Path] = []
    pending_benchmarks: list[str] = []
    for benchmark, summary in benchmark_summaries.items():
        if summary.get("status") == "disabled":
            continue
        if not summary.get("include_in_training_dataset", True):
            continue
        if summary.get("status") != "completed":
            pending_benchmarks.append(benchmark)
            continue
        verified_output = summary.get("verified_output")
        corrupted_main_output = summary.get("corrupted_main_output")
        if not verified_output or not corrupted_main_output:
            pending_benchmarks.append(benchmark)
            continue
        clean_paths.append(Path(verified_output))
        corrupted_paths.append(Path(corrupted_main_output))

    if pending_benchmarks or not clean_paths or not corrupted_paths:
        return {
            "status": "skipped",
            "reason": "missing_completed_benchmarks",
            "pending_benchmarks": pending_benchmarks,
        }

    clean_traces: list[TraceRecord] = []
    corrupted_traces: list[TraceRecord] = []
    for path in clean_paths:
        clean_traces.extend(load_traces(path))
    for path in corrupted_paths:
        corrupted_traces.extend(load_traces(path))

    pointwise = build_pointwise_dataset(
        clean_traces,
        corrupted_traces,
        critical_threshold=float(training_cfg.get("critical_threshold", 0.5)),
        critical_penalty=float(training_cfg.get("critical_penalty", 0.5)),
    )
    pairwise = build_pairwise_dataset(clean_traces, corrupted_traces)

    pointwise_output = _resolve_path(training_cfg["pointwise_output"])
    pairwise_output = _resolve_path(training_cfg["pairwise_output"])
    assert pointwise_output is not None and pairwise_output is not None
    _ensure_parent(pointwise_output)
    _ensure_parent(pairwise_output)
    write_pointwise_dataset(pointwise_output, pointwise)
    write_pairwise_dataset(pairwise_output, pairwise)

    return {
        "status": "completed",
        "pointwise_count": len(pointwise),
        "pairwise_count": len(pairwise),
        "pointwise_output": str(pointwise_output),
        "pairwise_output": str(pairwise_output),
    }


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full available pipeline around externally supplied teacher outputs."""
    teacher_cfg = dict(config.get("teacher") or {})
    if "model_name" not in teacher_cfg:
        raise ValueError("Pipeline config must include `teacher.model_name`.")

    benchmark_summaries: dict[str, dict[str, Any]] = {}
    benchmark_items = dict(config.get("benchmarks") or {})
    base_seed = int(teacher_cfg.get("seed", 0))

    for benchmark, benchmark_cfg in benchmark_items.items():
        if not benchmark_cfg.get("enabled", True):
            benchmark_summaries[benchmark] = {"status": "disabled"}
            continue

        summary: dict[str, Any] = {
            "status": "started",
            "include_in_training_dataset": bool(
                benchmark_cfg.get("include_in_training_dataset", benchmark != "visualwebbench")
            ),
        }
        manifest, manifest_path = _build_manifest(benchmark, benchmark_cfg)
        summary["manifest_output"] = str(manifest_path)
        summary["manifest_count"] = len(manifest)

        requests, requests_path = _prepare_requests(
            manifest_path,
            benchmark,
            benchmark_cfg,
            teacher_cfg,
        )
        summary["teacher_requests_output"] = str(requests_path)
        summary["teacher_request_count"] = len(requests)

        teacher_outputs_input = _resolve_path(benchmark_cfg.get("teacher_outputs_input"))
        if teacher_outputs_input is None or not teacher_outputs_input.exists():
            summary["status"] = "waiting_for_teacher_outputs"
            summary["missing_teacher_outputs"] = (
                None if teacher_outputs_input is None else str(teacher_outputs_input)
            )
            benchmark_summaries[benchmark] = summary
            continue

        parsed_output = _resolve_path(benchmark_cfg["teacher_traces_output"])
        verified_output = _resolve_path(benchmark_cfg["verified_output"])
        rejected_output = _resolve_path(benchmark_cfg["rejected_output"])
        assert parsed_output is not None and verified_output is not None and rejected_output is not None

        parsed_traces = _parse_teacher_outputs(teacher_outputs_input, parsed_output)
        verified, rejected = _verify_traces(
            manifest_path,
            parsed_traces,
            verified_output_path=verified_output,
            rejected_output_path=rejected_output,
        )
        corruption_counts = _build_corruptions_for_benchmark(
            manifest_path,
            verified,
            benchmark_cfg,
            base_seed=base_seed,
        )
        summary.update(
            {
                "status": "completed",
                "teacher_outputs_input": str(teacher_outputs_input),
                "parsed_output": str(parsed_output),
                "parsed_count": len(parsed_traces),
                "verified_output": str(verified_output),
                "verified_count": len(verified),
                "rejected_output": str(rejected_output),
                "rejected_count": len(rejected),
                "corrupted_main_output": str(_resolve_path(benchmark_cfg["corrupted_main_output"])),
                "corrupted_cross_output": str(_resolve_path(benchmark_cfg["corrupted_cross_output"])),
                "corrupted_wrong_use_output": str(
                    _resolve_path(benchmark_cfg["corrupted_wrong_use_output"])
                ),
                "corruption_counts": corruption_counts,
            }
        )
        benchmark_summaries[benchmark] = summary

    training_summary = _maybe_build_training_dataset(config, benchmark_summaries)
    result = {
        "benchmarks": benchmark_summaries,
        "training_dataset": training_summary,
    }
    summary_output = _resolve_path(config.get("summary_output"))
    if summary_output is not None:
        _ensure_parent(summary_output)
        summary_output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        result["summary_output"] = str(summary_output)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CG-PRM pipeline from benchmark normalization to training-data export. "
            "Teacher inference itself remains external: provide teacher output JSONL paths in the config "
            "to unlock downstream stages."
        )
    )
    parser.add_argument("--config", required=True, help="Path to a pipeline JSON config.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = _read_config(args.config)
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
