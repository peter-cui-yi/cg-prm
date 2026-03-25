"""Build training-ready supervision datasets from verified and corrupted traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from cg_prm.data.schema import TraceRecord, read_jsonl, write_jsonl
from cg_prm.evaluation.reranking import (
    aggregate_step_scores,
    infer_grounding_critical_mask,
)


@dataclass(frozen=True, slots=True)
class PointwiseTrainingExample:
    """Pointwise supervision record for one full trace."""

    record_id: str
    example_id: str
    benchmark: str
    trace_id: str
    dataset_role: str
    trace_label: int
    trace_score_target: float
    answer_correct: bool | None
    corruption_family: str | None
    wrong_use_type: str | None
    source_trace_id: str | None
    step_labels: list[int]
    step_error_types: list[str]
    critical_step_mask: list[int]
    trace: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "example_id": self.example_id,
            "benchmark": self.benchmark,
            "trace_id": self.trace_id,
            "dataset_role": self.dataset_role,
            "trace_label": self.trace_label,
            "trace_score_target": self.trace_score_target,
            "answer_correct": self.answer_correct,
            "corruption_family": self.corruption_family,
            "wrong_use_type": self.wrong_use_type,
            "source_trace_id": self.source_trace_id,
            "step_labels": self.step_labels,
            "step_error_types": self.step_error_types,
            "critical_step_mask": self.critical_step_mask,
            "trace": self.trace,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PairwiseTrainingExample:
    """Pairwise supervision record for ranking one preferred trace over one rejected trace."""

    pair_id: str
    example_id: str
    benchmark: str
    pair_type: str
    preferred_trace_id: str
    rejected_trace_id: str
    corruption_family: str | None
    wrong_use_type: str | None
    preferred_trace: dict[str, Any]
    rejected_trace: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "example_id": self.example_id,
            "benchmark": self.benchmark,
            "pair_type": self.pair_type,
            "preferred_trace_id": self.preferred_trace_id,
            "rejected_trace_id": self.rejected_trace_id,
            "corruption_family": self.corruption_family,
            "wrong_use_type": self.wrong_use_type,
            "preferred_trace": self.preferred_trace,
            "rejected_trace": self.rejected_trace,
            "metadata": self.metadata,
        }


def load_traces(path: str | Path) -> list[TraceRecord]:
    """Load trace records from JSONL."""
    return [TraceRecord.from_dict(payload) for payload in read_jsonl(path)]


def _trace_answer_correct(trace: TraceRecord) -> bool | None:
    validation = trace.metadata.get("validation")
    if isinstance(validation, Mapping) and "answer_correct" in validation:
        answer_correct = validation.get("answer_correct")
        if answer_correct is None:
            return None
        return bool(answer_correct)
    if trace.predicted_answer is None:
        return None
    return str(trace.predicted_answer).strip().lower() == str(trace.gold_answer).strip().lower()


def _trace_role(trace: TraceRecord) -> str:
    return "corrupted" if isinstance(trace.metadata.get("corruption"), Mapping) else "clean"


def _trace_corruption_info(trace: TraceRecord) -> dict[str, Any]:
    payload = trace.metadata.get("corruption")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _pointwise_example_from_trace(
    trace: TraceRecord,
    *,
    critical_threshold: float,
    critical_penalty: float,
) -> PointwiseTrainingExample:
    step_labels = [int(step.label) for step in trace.steps]
    critical_mask = [1 if flag else 0 for flag in infer_grounding_critical_mask(trace)]
    trace_score = aggregate_step_scores(
        step_scores=[float(label) for label in step_labels],
        critical_mask=[bool(flag) for flag in critical_mask],
        threshold=critical_threshold,
        critical_penalty=critical_penalty,
    )
    corruption = _trace_corruption_info(trace)
    return PointwiseTrainingExample(
        record_id=f"pointwise::{trace.trace_id}",
        example_id=trace.example_id,
        benchmark=trace.benchmark,
        trace_id=trace.trace_id,
        dataset_role=_trace_role(trace),
        trace_label=int(all(label == 1 for label in step_labels)),
        trace_score_target=trace_score,
        answer_correct=_trace_answer_correct(trace),
        corruption_family=corruption.get("family"),
        wrong_use_type=corruption.get("wrong_use_type"),
        source_trace_id=corruption.get("source_trace_id"),
        step_labels=step_labels,
        step_error_types=[step.error_type for step in trace.steps],
        critical_step_mask=critical_mask,
        trace=trace.to_dict(),
        metadata={
            "trace_mode": trace.trace_mode,
            "num_steps": len(trace.steps),
        },
    )


def build_pointwise_dataset(
    clean_traces: Iterable[TraceRecord],
    corrupted_traces: Iterable[TraceRecord] = (),
    *,
    critical_threshold: float = 0.5,
    critical_penalty: float = 0.5,
) -> list[PointwiseTrainingExample]:
    """Build pointwise trace supervision examples from clean and corrupted traces."""
    examples: list[PointwiseTrainingExample] = []
    for trace in list(clean_traces) + list(corrupted_traces):
        examples.append(
            _pointwise_example_from_trace(
                trace,
                critical_threshold=critical_threshold,
                critical_penalty=critical_penalty,
            )
        )
    return examples


def _index_clean_traces(clean_traces: Iterable[TraceRecord]) -> tuple[dict[str, TraceRecord], dict[str, list[TraceRecord]]]:
    by_trace_id: dict[str, TraceRecord] = {}
    by_example_id: dict[str, list[TraceRecord]] = {}
    for trace in clean_traces:
        by_trace_id[trace.trace_id] = trace
        by_example_id.setdefault(trace.example_id, []).append(trace)
    return by_trace_id, by_example_id


def _match_preferred_trace(
    corrupted_trace: TraceRecord,
    *,
    clean_by_trace_id: Mapping[str, TraceRecord],
    clean_by_example_id: Mapping[str, list[TraceRecord]],
) -> TraceRecord | None:
    corruption = _trace_corruption_info(corrupted_trace)
    source_trace_id = corruption.get("source_trace_id")
    if source_trace_id and source_trace_id in clean_by_trace_id:
        return clean_by_trace_id[source_trace_id]
    candidates = clean_by_example_id.get(corrupted_trace.example_id, [])
    if not candidates:
        return None
    return candidates[0]


def build_pairwise_dataset(
    clean_traces: Iterable[TraceRecord],
    corrupted_traces: Iterable[TraceRecord],
) -> list[PairwiseTrainingExample]:
    """Build pairwise ranking supervision from clean-vs-corrupted trace pairs."""
    clean_list = list(clean_traces)
    corrupted_list = list(corrupted_traces)
    clean_by_trace_id, clean_by_example_id = _index_clean_traces(clean_list)

    pairs: list[PairwiseTrainingExample] = []
    for corrupted_trace in corrupted_list:
        preferred_trace = _match_preferred_trace(
            corrupted_trace,
            clean_by_trace_id=clean_by_trace_id,
            clean_by_example_id=clean_by_example_id,
        )
        if preferred_trace is None:
            continue
        corruption = _trace_corruption_info(corrupted_trace)
        pair_id = (
            f"pair::{preferred_trace.trace_id}::"
            f"{corrupted_trace.trace_id}"
        )
        pairs.append(
            PairwiseTrainingExample(
                pair_id=pair_id,
                example_id=corrupted_trace.example_id,
                benchmark=corrupted_trace.benchmark,
                pair_type="clean_vs_corrupted",
                preferred_trace_id=preferred_trace.trace_id,
                rejected_trace_id=corrupted_trace.trace_id,
                corruption_family=corruption.get("family"),
                wrong_use_type=corruption.get("wrong_use_type"),
                preferred_trace=preferred_trace.to_dict(),
                rejected_trace=corrupted_trace.to_dict(),
                metadata={
                    "source_trace_id": corruption.get("source_trace_id"),
                    "generator_name": corruption.get("generator_name"),
                    "preferred_answer_correct": _trace_answer_correct(preferred_trace),
                    "rejected_answer_correct": _trace_answer_correct(corrupted_trace),
                },
            )
        )
    return pairs


def write_pointwise_dataset(path: str | Path, examples: Iterable[PointwiseTrainingExample]) -> None:
    """Write pointwise supervision records to JSONL."""
    write_jsonl(path, examples)


def write_pairwise_dataset(path: str | Path, examples: Iterable[PairwiseTrainingExample]) -> None:
    """Write pairwise supervision records to JSONL."""
    write_jsonl(path, examples)
