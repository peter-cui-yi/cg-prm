"""Step-level evaluation metrics for CG-PRM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepMetrics:
    """Metrics for step-level error detection."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_samples: int


def compute_step_detection_metrics(
    predictions: list[int],
    labels: list[int],
) -> StepMetrics:
    """Compute metrics for step-level error detection.
    
    Args:
        predictions: Binary predictions (1=error detected, 0=no error)
        labels: Ground truth labels (1=error present, 0=no error)
    
    Returns:
        StepMetrics with accuracy, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return StepMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        num_samples=len(labels),
    )


def analyze_first_divergence_detection(
    test_pairs: list[dict[str, Any]],
    model_predictions: list[tuple[float, float]],
) -> dict[str, Any]:
    """Analyze model's ability to detect first divergence point (t_star).
    
    Args:
        test_pairs: List of test pairs with t_star annotations
        model_predictions: List of (pos_score, neg_score) tuples
    
    Returns:
        Dictionary with t_star detection metrics
    """
    results = {
        "correct_detections": 0,
        "total": len(test_pairs),
        "by_t_star": {},
        "by_family": {},
    }
    
    t_star_correct = {}
    family_correct = {}
    
    for pair, (pos_score, neg_score) in zip(test_pairs, model_predictions):
        t_star = pair.get("t_star", 1)
        family = pair.get("family", "unknown")
        
        # Model correctly identifies positive trace
        correct = pos_score > neg_score
        
        if correct:
            results["correct_detections"] += 1
        
        # Track by t_star
        if t_star not in t_star_correct:
            t_star_correct[t_star] = {"correct": 0, "total": 0}
        t_star_correct[t_star]["total"] += 1
        if correct:
            t_star_correct[t_star]["correct"] += 1
        
        # Track by family
        if family not in family_correct:
            family_correct[family] = {"correct": 0, "total": 0}
        family_correct[family]["total"] += 1
        if correct:
            family_correct[family]["correct"] += 1
    
    # Compute accuracy by t_star
    for t_star, counts in sorted(t_star_correct.items()):
        results["by_t_star"][str(t_star)] = {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0,
            "num_samples": counts["total"],
        }
    
    # Compute accuracy by family
    for family, counts in sorted(family_correct.items()):
        results["by_family"][family] = {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0,
            "num_samples": counts["total"],
        }
    
    results["overall_accuracy"] = results["correct_detections"] / results["total"] if results["total"] > 0 else 0.0
    
    return results


def compute_calibration_metrics(
    scores: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration metrics.
    
    Args:
        scores: Model confidence scores (0-1)
        labels: Binary labels (1=correct, 0=incorrect)
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with calibration metrics
    """
    import numpy as np
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Bin scores
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(scores, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute calibration error
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(scores[mask])
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(np.sum(mask))
    
    # Expected Calibration Error (ECE)
    total = sum(bin_counts)
    ece = sum(
        (count / total) * abs(acc - conf)
        for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
    )
    
    # Maximum Calibration Error (MCE)
    mce = max(abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences))
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "num_bins": n_bins,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }
