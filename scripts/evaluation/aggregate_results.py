#!/usr/bin/env python3
"""Aggregate all evaluation results into comprehensive report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_results(input_dir: Path) -> dict[str, Any]:
    """Load all results from evaluation runs."""
    
    results = {}
    
    # Load main results
    main_results_path = input_dir / "full_results.json"
    if main_results_path.exists():
        results["main"] = json.loads(main_results_path.read_text())
    
    # Load ablation results
    ablation_path = input_dir / "corruption_ablation.json"
    if ablation_path.exists():
        results["ablation"] = json.loads(ablation_path.read_text())
    
    # Load step analysis
    step_path = input_dir / "step_analysis.json"
    if step_path.exists():
        results["step_analysis"] = json.loads(step_path.read_text())
    
    return results


def compute_statistical_significance(
    auroc1: float,
    ci1: list[float],
    auroc2: float,
    ci2: list[float],
) -> dict[str, Any]:
    """Compute statistical significance of difference."""
    
    # Simple check: non-overlapping CIs
    significant = ci1[0] > ci2[1] or ci1[1] < ci2[0]
    
    return {
        "significant": significant,
        "method": "non_overlapping_ci",
        "delta": auroc1 - auroc2,
    }


def generate_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Generate comprehensive summary."""
    
    main = results.get("main", {})
    ablation = results.get("ablation", {})
    step_analysis = results.get("step_analysis", {})
    
    summary = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "experiment": "full_scale_cg_prm",
        },
        "main_results": {
            "cg_prm_auroc": main.get("cg_prm", {}).get("auroc"),
            "cg_prm_ci_95": main.get("cg_prm", {}).get("ci_95"),
            "pointwise_auroc": main.get("pointwise", {}).get("auroc"),
            "pointwise_ci_95": main.get("pointwise", {}).get("ci_95"),
            "delta": main.get("delta"),
            "decision": main.get("decision"),
        },
        "statistical_significance": compute_statistical_significance(
            main.get("cg_prm", {}).get("auroc", 0),
            main.get("cg_prm", {}).get("ci_95", [0, 0]),
            main.get("pointwise", {}).get("auroc", 0),
            main.get("pointwise", {}).get("ci_95", [0, 0]),
        ),
        "corruption_ablation": ablation,
        "step_level_analysis": step_analysis,
        "conclusion": generate_conclusion(main, ablation, step_analysis),
    }
    
    return summary


def generate_conclusion(
    main: dict[str, Any],
    ablation: dict[str, Any],
    step_analysis: dict[str, Any],
) -> str:
    """Generate natural language conclusion."""
    
    cg_auroc = main.get("cg_prm", {}).get("auroc", 0)
    pw_auroc = main.get("pointwise", {}).get("auroc", 0)
    delta = main.get("delta", 0)
    decision = main.get("decision", "UNKNOWN")
    
    parts = []
    
    # Main finding
    if decision == "GO":
        parts.append(f"CG-PRM (AUROC={cg_auroc:.4f}) significantly outperforms pointwise baseline (AUROC={pw_auroc:.4f}), with delta={delta:.4f}.")
    else:
        parts.append(f"CG-PRM (AUROC={cg_auroc:.4f}) does not show significant improvement over pointwise baseline (AUROC={pw_auroc:.4f}), with delta={delta:.4f}.")
    
    # Best corruption family
    if ablation:
        best_family = max(
            [(k, v.get("auroc", 0)) for k, v in ablation.items() if v.get("auroc")],
            key=lambda x: x[1],
            default=(None, 0)
        )
        if best_family[0]:
            parts.append(f"Best performance on {best_family[0]} corruption (AUROC={best_family[1]:.4f}).")
    
    # Step detection
    if step_analysis:
        step_acc = step_analysis.get("overall_accuracy", 0)
        parts.append(f"Step-level error detection accuracy: {step_acc:.4f}.")
    
    return " ".join(parts)


def generate_latex_tables(summary: dict[str, Any]) -> str:
    """Generate LaTeX tables for paper."""
    
    main = summary["main_results"]
    
    # Main results table
    main_table = f"""\\begin{{table}}[t]
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Model}} & \\textbf{{AUROC}} & \\textbf{{95\\% CI}} \\\\
\\midrule
Pointwise & {main['pointwise_auroc']:.4f} & [{main['pointwise_ci_95'][0]:.4f}, {main['pointwise_ci_95'][1]:.4f}] \\\\
CG-PRM (Ours) & \\textbf{{{main['cg_prm_auroc']:.4f}}} & [{main['cg_prm_ci_95'][0]:.4f}, {main['cg_prm_ci_95'][1]:.4f}] \\\\
\\midrule
Delta & \\textbf{{{main['delta']:.4f}}} & \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Main results: CG-PRM vs Pointwise baseline}}
\\end{{table}}"""
    
    return main_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--input_dir", required=True, help="Input directory with results")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--generate-latex", action="store_true", help="Generate LaTeX tables")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Aggregating Full-Scale Experiment Results")
    print("=" * 60)
    print("")
    
    # Load results
    print(f"Loading results from {args.input_dir}...")
    results = load_results(Path(args.input_dir))
    print(f"Loaded: {list(results.keys())}")
    print("")
    
    # Generate summary
    print("Generating summary...")
    summary = generate_summary(results)
    print("")
    
    # Print conclusion
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(summary["conclusion"])
    print("")
    
    # Generate LaTeX if requested
    if args.generate_latex:
        latex_tables = generate_latex_tables(summary)
        output_dir = Path(args.output).parent
        latex_path = output_dir / "main_results_table.tex"
        latex_path.write_text(latex_tables)
        print(f"LaTeX tables saved to: {latex_path}")
        print("")
    
    # Save summary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {output_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
