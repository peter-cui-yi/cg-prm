#!/usr/bin/env python3
"""CLI for rendering teacher requests from normalized manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_prm.data.manifests import load_manifest
from cg_prm.data.schema import write_jsonl
from cg_prm.generation.teacher import GenerationConfig, build_teacher_requests


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render teacher generation requests from a CG-PRM manifest."
    )
    parser.add_argument("--manifest", required=True, help="Normalized example manifest JSONL.")
    parser.add_argument("--output", required=True, help="Destination request JSONL.")
    parser.add_argument("--model", required=True, help="Teacher model name.")
    parser.add_argument("--prompt-id", required=True, help="Prompt template id.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", choices=("docvqa", "clevr"))
    parser.add_argument("--limit", type=int, help="Optional maximum number of examples.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    examples = load_manifest(args.manifest)
    if args.benchmark:
        examples = [example for example in examples if example.benchmark == args.benchmark]
    if args.limit is not None:
        examples = examples[: args.limit]

    config = GenerationConfig(
        model_name=args.model,
        prompt_id=args.prompt_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    requests = build_teacher_requests(examples, config)
    write_jsonl(args.output, requests)
    print(f"Built {len(requests)} teacher requests at {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
