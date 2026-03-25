"""Evaluation metrics and reranking helpers for CG-PRM."""

from cg_prm.evaluation.metrics import (
    average_precision,
    bootstrap_confidence_interval,
    false_acceptance_rate,
    paired_ranking_accuracy,
    roc_auc,
)
from cg_prm.evaluation.reranking import (
    ScoredTrace,
    aggregate_step_scores,
    infer_grounding_critical_mask,
    rank_traces,
    rerank_groups,
    score_trace_with_step_scores,
    select_best_under_budget,
)

__all__ = [
    "ScoredTrace",
    "aggregate_step_scores",
    "average_precision",
    "bootstrap_confidence_interval",
    "false_acceptance_rate",
    "infer_grounding_critical_mask",
    "paired_ranking_accuracy",
    "rank_traces",
    "rerank_groups",
    "roc_auc",
    "score_trace_with_step_scores",
    "select_best_under_budget",
]
