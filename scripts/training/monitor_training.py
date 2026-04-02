#!/usr/bin/env python3
"""Monitor training progress and plot metrics in real-time."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_training_log(log_path: Path) -> dict:
    """Parse training log to extract metrics."""
    
    metrics = {
        "steps": [],
        "train_loss": [],
        "eval_loss": [],
        "eval_auroc": [],
        "learning_rate": [],
    }
    
    with open(log_path) as f:
        for line in f:
            # Extract step
            step_match = re.search(r"step=(\d+)", line)
            if step_match:
                step = int(step_match.group(1))
                metrics["steps"].append(step)
            
            # Extract loss
            loss_match = re.search(r"loss=(\d+\.\d+)", line)
            if loss_match:
                metrics["train_loss"].append(float(loss_match.group(1)))
            
            # Extract eval loss
            eval_loss_match = re.search(r"eval_loss=(\d+\.\d+)", line)
            if eval_loss_match:
                metrics["eval_loss"].append(float(eval_loss_match.group(1)))
            
            # Extract learning rate
            lr_match = re.search(r"learning_rate=(\d+\.?\d*e?-?\d*)", line)
            if lr_match:
                metrics["learning_rate"].append(float(lr_match.group(1)))
    
    return metrics


def plot_training_curves(metrics: dict, output_path: Path) -> None:
    """Plot training curves."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if metrics["train_loss"]:
        axes[0, 0].plot(metrics["steps"], metrics["train_loss"], label="Train Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Learning rate
    if metrics["learning_rate"]:
        axes[0, 1].plot(metrics["steps"], metrics["learning_rate"], label="Learning Rate")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("LR")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Evaluation loss
    if metrics["eval_loss"]:
        axes[1, 0].plot(metrics["steps"][:len(metrics["eval_loss"])], metrics["eval_loss"], label="Eval Loss")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Evaluation Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Hide empty subplot
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Training curves saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--log_dir", default="logs", help="Directory with training logs")
    parser.add_argument("--output", default="logs/training_curves.png", help="Output plot path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Monitor")
    print("=" * 60)
    print("")
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        return 1
    
    # Find training logs
    train_logs = list(log_dir.glob("*train.log"))
    if not train_logs:
        print(f"ERROR: No training logs found in {log_dir}")
        return 1
    
    print(f"Found {len(train_logs)} training logs:")
    for log in train_logs:
        print(f"  - {log.name}")
    print("")
    
    # Parse and plot each log
    for log_path in train_logs:
        print(f"Parsing {log_path.name}...")
        metrics = parse_training_log(log_path)
        
        if not metrics["train_loss"]:
            print(f"  WARNING: No metrics found in {log_path.name}")
            continue
        
        print(f"  Steps: {len(metrics['steps'])}")
        print(f"  Train loss points: {len(metrics['train_loss'])}")
        print(f"  Eval loss points: {len(metrics['eval_loss'])}")
        
        # Plot
        output_name = log_path.stem.replace("_train", "_curves") + ".png"
        output_path = Path(args.output).parent / output_name
        plot_training_curves(metrics, output_path)
    
    print("")
    print("=" * 60)
    print("Monitoring complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
