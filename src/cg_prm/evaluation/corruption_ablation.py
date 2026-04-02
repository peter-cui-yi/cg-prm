"""Corruption family ablation analysis for CG-PRM."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    scores_pos: list[float],
    scores_neg: list[float],
    n_bootstrap: int = 1000,
    ci: float = 95.0,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for AUROC."""
    aurocs = []
    rng = np.random.default_rng(42)
    n = len(scores_pos)
    
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sp = [scores_pos[i] for i in idx]
        sn = [scores_neg[i] for i in idx]
        labels = [1] * n + [0] * n
        scores = sp + sn
        if len(set(labels)) > 1:
            aurocs.append(roc_auc_score(labels, scores))
    
    if not aurocs:
        return 0.5, 0.0, 1.0
    
    alpha = 100 - ci
    return (
        float(np.mean(aurocs)),
        float(np.percentile(aurocs, alpha / 2)),
        float(np.percentile(aurocs, 100 - alpha / 2)),
    )


def analyze_by_corruption_family(
    test_pairs: list[dict[str, Any]],
    model_scores: list[tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    """Analyze model performance by corruption family.
    
    Args:
        test_pairs: Test pairs with family annotations
        model_scores: List of (pos_score, neg_score) tuples
    
    Returns:
        Dictionary with metrics per corruption family
    """
    family_data = {}
    
    # Group by family
    for pair, (pos_score, neg_score) in zip(test_pairs, model_scores):
        family = pair.get("family", "unknown")
        
        if family not in family_data:
            family_data[family] = {
                "pos_scores": [],
                "neg_scores": [],
                "correct": 0,
                "total": 0,
            }
        
        family_data[family]["pos_scores"].append(pos_score)
        family_data[family]["neg_scores"].append(neg_score)
        family_data[family]["total"] += 1
        if pos_score > neg_score:
            family_data[family]["correct"] += 1
    
    # Compute metrics per family
    results = {}
    for family, data in family_data.items():
        accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        
        if len(data["pos_scores"]) > 1:
            auroc, ci_lower, ci_upper = bootstrap_ci(
                data["pos_scores"], data["neg_scores"]
            )
            results[family] = {
                "auroc": auroc,
                "ci_95": [ci_lower, ci_upper],
                "accuracy": accuracy,
                "num_samples": data["total"],
                "mean_pos_score": float(np.mean(data["pos_scores"])),
                "mean_neg_score": float(np.mean(data["neg_scores"])),
                "std_pos_score": float(np.std(data["pos_scores"])),
                "std_neg_score": float(np.std(data["neg_scores"])),
            }
        else:
            results[family] = {
                "auroc": None,
                "accuracy": accuracy,
                "num_samples": data["total"],
            }
    
    return results


def identify_best_worst_families(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Identify best and worst performing corruption families.
    
    Args:
        results: Output from analyze_by_corruption_family
    
    Returns:
        Dictionary with best/worst families
    """
    families_with_auroc = [
        (family, data["auroc"])
        for family, data in results.items()
        if data["auroc"] is not None
    ]
    
    if not families_with_auroc:
        return {
            "best": None,
            "worst": None,
            "note": "No families with AUROC computed",
        }
    
    best_family = max(families_with_auroc, key=lambda x: x[1])
    worst_family = min(families_with_auroc, key=lambda x: x[1])
    
    return {
        "best": {
            "family": best_family[0],
            "auroc": best_family[1],
            "ci_95": results[best_family[0]]["ci_95"],
            "num_samples": results[best_family[0]]["num_samples"],
        },
        "worst": {
            "family": worst_family[0],
            "auroc": worst_family[1],
            "ci_95": results[worst_family[0]]["ci_95"],
            "num_samples": results[worst_family[0]]["num_samples"],
        },
    }
