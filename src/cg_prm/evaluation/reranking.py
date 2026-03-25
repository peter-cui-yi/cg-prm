"""Trace scoring and reranking helpers for verifier evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from cg_prm.data.schema import TraceRecord

GROUNDING_CRITICAL_STEP_TYPES = frozenset(
    {"locate", "read", "extract", "identify", "relate", "count", "compute", "reason", "verify"}
)


@dataclass(frozen=True, slots=True)
class ScoredTrace:
    """Trace plus step-level and aggregated verifier scores."""

    trace: TraceRecord
    step_scores: list[float]
    final_score: float
    critical_mask: list[bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace.trace_id,
            "example_id": self.trace.example_id,
            "benchmark": self.trace.benchmark,
            "step_scores": self.step_scores,
            "final_score": self.final_score,
            "critical_mask": self.critical_mask,
        }


def infer_grounding_critical_mask(trace: TraceRecord) -> list[bool]:
    """Infer which steps should trigger extra penalties if they fail."""
    return [step.step_type in GROUNDING_CRITICAL_STEP_TYPES for step in trace.steps]


def aggregate_step_scores(
    step_scores: Sequence[float],
    *,
    critical_mask: Sequence[bool] | None = None,
    threshold: float = 0.5,
    critical_penalty: float = 0.5,
) -> float:
    """Aggregate step scores with a penalty for failed grounding-critical steps."""
    if not step_scores:
        return 0.0
    if critical_mask is None:
        critical_mask = [True] * len(step_scores)
    if len(step_scores) != len(critical_mask):
        raise ValueError("`step_scores` and `critical_mask` must have the same length.")

    mean_score = sum(step_scores) / len(step_scores)
    failed_critical = any(
        is_critical and score < threshold
        for score, is_critical in zip(step_scores, critical_mask)
    )
    if failed_critical:
        mean_score -= critical_penalty
    return max(0.0, mean_score)


def score_trace_with_step_scores(
    trace: TraceRecord,
    step_scores: Sequence[float],
    *,
    threshold: float = 0.5,
    critical_penalty: float = 0.5,
) -> ScoredTrace:
    """Build a scored trace object from external per-step predictions."""
    critical_mask = infer_grounding_critical_mask(trace)
    final_score = aggregate_step_scores(
        step_scores,
        critical_mask=critical_mask,
        threshold=threshold,
        critical_penalty=critical_penalty,
    )
    return ScoredTrace(
        trace=trace,
        step_scores=list(step_scores),
        final_score=final_score,
        critical_mask=critical_mask,
    )


def _resolve_trace_score(
    trace: TraceRecord,
    *,
    score_lookup: Mapping[str, float] | None = None,
    scorer: Callable[[TraceRecord], float] | None = None,
) -> float:
    if scorer is not None:
        return float(scorer(trace))
    if score_lookup is None:
        raise ValueError("Either `score_lookup` or `scorer` must be provided.")
    return float(score_lookup[trace.trace_id])


def rank_traces(
    traces: Iterable[TraceRecord],
    *,
    score_lookup: Mapping[str, float] | None = None,
    scorer: Callable[[TraceRecord], float] | None = None,
    descending: bool = True,
) -> list[tuple[TraceRecord, float]]:
    """Sort traces by score."""
    scored = [
        (trace, _resolve_trace_score(trace, score_lookup=score_lookup, scorer=scorer))
        for trace in traces
    ]
    return sorted(scored, key=lambda item: item[1], reverse=descending)


def select_best_under_budget(
    traces: Sequence[TraceRecord],
    *,
    budget: float,
    score_lookup: Mapping[str, float] | None = None,
    scorer: Callable[[TraceRecord], float] | None = None,
    cost_lookup: Mapping[str, float] | None = None,
    cost_fn: Callable[[TraceRecord], float] | None = None,
) -> tuple[TraceRecord | None, list[TraceRecord]]:
    """Select the best-scoring trace among the prefix that fits a fixed budget."""
    if budget <= 0 or not traces:
        return None, []

    observed: list[TraceRecord] = []
    spent = 0.0
    for trace in traces:
        if cost_fn is not None:
            cost = float(cost_fn(trace))
        elif cost_lookup is not None:
            cost = float(cost_lookup.get(trace.trace_id, 1.0))
        else:
            cost = 1.0
        if spent + cost > budget and observed:
            break
        if spent + cost > budget and not observed:
            return None, []
        observed.append(trace)
        spent += cost

    if not observed:
        return None, []
    ranked = rank_traces(observed, score_lookup=score_lookup, scorer=scorer)
    return ranked[0][0], observed


def rerank_groups(
    candidate_groups: Mapping[str, Sequence[TraceRecord]],
    *,
    score_lookup: Mapping[str, float] | None = None,
    scorer: Callable[[TraceRecord], float] | None = None,
    budget: float | None = None,
    cost_lookup: Mapping[str, float] | None = None,
    cost_fn: Callable[[TraceRecord], float] | None = None,
) -> dict[str, TraceRecord]:
    """Rerank one candidate group per example and return the selected traces."""
    selected: dict[str, TraceRecord] = {}
    for group_id, traces in candidate_groups.items():
        if budget is None:
            ranked = rank_traces(traces, score_lookup=score_lookup, scorer=scorer)
            if ranked:
                selected[group_id] = ranked[0][0]
            continue

        best_trace, _ = select_best_under_budget(
            traces,
            budget=budget,
            score_lookup=score_lookup,
            scorer=scorer,
            cost_lookup=cost_lookup,
            cost_fn=cost_fn,
        )
        if best_trace is not None:
            selected[group_id] = best_trace
    return selected
