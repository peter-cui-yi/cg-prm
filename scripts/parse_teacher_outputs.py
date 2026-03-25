#!/usr/bin/env python3
"""CLI for parsing raw teacher outputs into structured traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_prm.data.schema import read_jsonl, write_jsonl
from cg_prm.generation.teacher import TeacherOutput, parse_teacher_output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse raw teacher outputs into structured CG-PRM traces."
    )
    parser.add_argument("--inputs", required=True, help="Raw teacher output JSONL.")
    parser.add_argument("--output", required=True, help="Destination trace JSONL.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    payloads = read_jsonl(args.inputs)
    traces = [parse_teacher_output(TeacherOutput.from_dict(payload)) for payload in payloads]
    write_jsonl(args.output, traces)
    print(f"Parsed {len(traces)} teacher outputs at {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
