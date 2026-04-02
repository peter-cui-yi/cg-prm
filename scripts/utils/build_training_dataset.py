#!/usr/bin/env python3
"""CLI for building pointwise and pairwise training datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_prm.training.dataset_builder import (
    build_pairwise_dataset,
    build_pointwise_dataset,
    load_traces,
    write_pairwise_dataset,
    write_pointwise_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build training-ready supervision files from clean and corrupted traces."
    )
    parser.add_argument("--clean", required=True, help="Verified clean trace JSONL.")
    parser.add_argument(
        "--corrupted",
        nargs="+",
        required=True,
        help="One or more corrupted trace JSONL files.",
    )
    parser.add_argument("--pointwise-output", required=True, help="Destination pointwise JSONL.")
    parser.add_argument("--pairwise-output", required=True, help="Destination pairwise JSONL.")
    parser.add_argument("--critical-threshold", type=float, default=0.5)
    parser.add_argument("--critical-penalty", type=float, default=0.5)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    clean_traces = load_traces(args.clean)
    corrupted_traces = []
    for path in args.corrupted:
        corrupted_traces.extend(load_traces(path))

    pointwise = build_pointwise_dataset(
        clean_traces,
        corrupted_traces,
        critical_threshold=args.critical_threshold,
        critical_penalty=args.critical_penalty,
    )
    pairwise = build_pairwise_dataset(clean_traces, corrupted_traces)

    write_pointwise_dataset(args.pointwise_output, pointwise)
    write_pairwise_dataset(args.pairwise_output, pairwise)
    print(
        f"Built {len(pointwise)} pointwise records and {len(pairwise)} pairwise records.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
