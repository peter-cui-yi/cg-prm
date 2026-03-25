"""Pure-Python evaluation metrics for verifier scoring and reranking."""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")


def safe_mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean or 0.0 for an empty iterable."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _validate_binary_labels(labels: Sequence[int]) -> None:
    invalid = [label for label in labels if label not in (0, 1)]
    if invalid:
        raise ValueError(f"Binary labels must be 0 or 1, got invalid values: {invalid[:5]}")


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC from binary labels and real-valued scores."""
    if len(labels) != len(scores):
        raise ValueError("`labels` and `scores` must have the same length.")
    if not labels:
        return 0.0
    _validate_binary_labels(labels)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    paired = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(paired):
        tie_end = index + 1
        while tie_end < len(paired) and paired[tie_end][0] == paired[index][0]:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        positive_count = sum(label for _, label in paired[index:tie_end])
        rank_sum += average_rank * positive_count
        index = tie_end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUPRC / average precision from binary labels and scores."""
    if len(labels) != len(scores):
        raise ValueError("`labels` and `scores` must have the same length.")
    if not labels:
        return 0.0
    _validate_binary_labels(labels)
    positives = sum(labels)
    if positives == 0:
        return 0.0

    paired = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    precision_sum = 0.0
    for _, label in paired:
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / (true_positives + false_positives)
        else:
            false_positives += 1
    return precision_sum / positives


def paired_ranking_accuracy(
    preferred_scores: Sequence[float],
    rejected_scores: Sequence[float],
    *,
    tie_value: float = 0.5,
) -> float:
    """Compute paired ranking accuracy for human challenge or pairwise evals."""
    if len(preferred_scores) != len(rejected_scores):
        raise ValueError("Score sequences must have the same length.")
    if not preferred_scores:
        return 0.0
    outcomes = []
    for preferred, rejected in zip(preferred_scores, rejected_scores):
        if preferred > rejected:
            outcomes.append(1.0)
        elif preferred < rejected:
            outcomes.append(0.0)
        else:
            outcomes.append(tie_value)
    return safe_mean(outcomes)


def false_acceptance_rate(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
    negative_label: int = 0,
) -> float:
    """Compute the rate at which negatives are incorrectly accepted."""
    if len(labels) != len(scores):
        raise ValueError("`labels` and `scores` must have the same length.")
    _validate_binary_labels(labels)
    negatives = [score for label, score in zip(labels, scores) if label == negative_label]
    if not negatives:
        return 0.0
    return sum(1 for score in negatives if score >= threshold) / len(negatives)


def top1_accuracy(labels: Sequence[int]) -> float:
    """Return the mean of binary top-1 outcomes."""
    _validate_binary_labels(labels)
    return safe_mean(float(label) for label in labels)


def coverage_rate(num_retained: int, num_total: int) -> float:
    """Compute retained coverage after filtering."""
    if num_total <= 0:
        return 0.0
    return num_retained / num_total


def bootstrap_confidence_interval(
    data: Sequence[T],
    metric_fn: Callable[[Sequence[T]], float],
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate mean metric and percentile bootstrap confidence interval."""
    if not data:
        return 0.0, 0.0, 0.0
    if n_bootstrap <= 0:
        raise ValueError("`n_bootstrap` must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("`alpha` must be between 0 and 1.")

    rng = random.Random(seed)
    point_estimate = metric_fn(data)
    samples: list[float] = []
    size = len(data)
    for _ in range(n_bootstrap):
        sample = [data[rng.randrange(size)] for _ in range(size)]
        samples.append(metric_fn(sample))
    samples.sort()
    lower_index = max(0, int(math.floor((alpha / 2.0) * n_bootstrap)) - 1)
    upper_index = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1)
    return point_estimate, samples[lower_index], samples[upper_index]
