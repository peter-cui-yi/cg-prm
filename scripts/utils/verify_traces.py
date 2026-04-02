#!/usr/bin/env python3
"""CLI for validating structured traces against normalized manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_prm.data.manifests import index_examples_by_id, load_manifest
from cg_prm.data.schema import TraceRecord, read_jsonl, write_jsonl
from cg_prm.verification.validators import annotate_trace_with_validation, validate_trace


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate structured traces and split them into verified and rejected outputs."
    )
    parser.add_argument("--manifest", required=True, help="Normalized example manifest JSONL.")
    parser.add_argument("--traces", required=True, help="Structured trace JSONL.")
    parser.add_argument("--verified-output", required=True, help="Destination verified trace JSONL.")
    parser.add_argument("--rejected-output", required=True, help="Destination rejected trace JSONL.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    example_lookup = index_examples_by_id(load_manifest(args.manifest))
    traces = [TraceRecord.from_dict(payload) for payload in read_jsonl(args.traces)]

    verified: list[TraceRecord] = []
    rejected: list[TraceRecord] = []
    for trace in traces:
        example = example_lookup.get(trace.example_id)
        if example is None:
            raise KeyError(f"Trace `{trace.trace_id}` refers to unknown example `{trace.example_id}`.")
        validation = validate_trace(example, trace)
        annotated = annotate_trace_with_validation(trace, validation)
        if validation.passed:
            verified.append(annotated)
        else:
            rejected.append(annotated)

    write_jsonl(args.verified_output, verified)
    write_jsonl(args.rejected_output, rejected)
    print(
        f"Validated {len(traces)} traces: {len(verified)} verified, {len(rejected)} rejected.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
