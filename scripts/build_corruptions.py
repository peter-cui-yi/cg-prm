#!/usr/bin/env python3
"""CLI for generating corrupted trace datasets from clean traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_prm.corruption import (
    CORRUPTION_FAMILIES,
    generate_corrupted_traces,
    generate_cross_corruptor_traces,
    generate_wrong_use_traces,
)
from cg_prm.data.manifests import index_examples_by_id, load_manifest
from cg_prm.data.schema import TraceRecord, read_jsonl, write_jsonl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build corrupted CG-PRM traces from clean structured traces."
    )
    parser.add_argument("--manifest", required=True, help="Normalized example manifest JSONL.")
    parser.add_argument("--traces", required=True, help="Clean trace JSONL.")
    parser.add_argument("--output", required=True, help="Destination corrupted trace JSONL.")
    parser.add_argument(
        "--mode",
        choices=("main", "cross", "wrong_use"),
        default="main",
        help="Corruption generator suite to run.",
    )
    parser.add_argument(
        "--families",
        help="Optional comma-separated subset of main corruption families.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    return parser


def _load_traces(path: str) -> list[TraceRecord]:
    return [TraceRecord.from_dict(payload) for payload in read_jsonl(path)]


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    examples = load_manifest(args.manifest)
    example_lookup = index_examples_by_id(examples)
    traces = _load_traces(args.traces)
    families = (
        [item.strip() for item in args.families.split(",") if item.strip()]
        if args.families
        else list(CORRUPTION_FAMILIES)
    )

    corrupted: list[TraceRecord] = []
    for index, trace in enumerate(traces):
        example = example_lookup.get(trace.example_id)
        if example is None:
            raise KeyError(f"Trace `{trace.trace_id}` refers to unknown example `{trace.example_id}`.")
        seed = args.seed + index
        if args.mode == "main":
            corrupted.extend(
                generate_corrupted_traces(example, trace, seed=seed, families=families)
            )
        elif args.mode == "cross":
            corrupted.extend(
                generate_cross_corruptor_traces(example, trace, seed=seed, families=families)
            )
        else:
            corrupted.extend(generate_wrong_use_traces(example, trace, seed=seed))

    write_jsonl(args.output, corrupted)
    print(
        f"Built {len(corrupted)} corrupted traces in {args.mode} mode at {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
