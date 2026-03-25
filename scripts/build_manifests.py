#!/usr/bin/env python3
"""CLI for building normalized DocVQA and CLEVR manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_prm.data.clevr import write_clevr_manifest
from cg_prm.data.docvqa import write_docvqa_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build normalized CG-PRM benchmark manifests."
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    docvqa = subparsers.add_parser("docvqa", help="Build a DocVQA manifest.")
    docvqa.add_argument("--questions", required=True, help="Path to the DocVQA questions JSON.")
    docvqa.add_argument("--images", required=True, help="Root directory containing image files.")
    docvqa.add_argument("--output", required=True, help="Destination JSONL manifest path.")
    docvqa.add_argument("--ocr", help="Optional OCR JSON path.")
    docvqa.add_argument("--split", help="Optional split name override.")

    clevr = subparsers.add_parser("clevr", help="Build a CLEVR manifest.")
    clevr.add_argument("--questions", required=True, help="Path to the CLEVR questions JSON.")
    clevr.add_argument("--images", required=True, help="Root directory containing image files.")
    clevr.add_argument("--output", required=True, help="Destination JSONL manifest path.")
    clevr.add_argument("--scenes", help="Optional CLEVR scenes JSON path.")
    clevr.add_argument("--split", help="Optional split name override.")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.benchmark == "docvqa":
        manifest = write_docvqa_manifest(
            output_path=args.output,
            questions_path=args.questions,
            image_root=args.images,
            ocr_path=args.ocr,
            split=args.split,
        )
    else:
        manifest = write_clevr_manifest(
            output_path=args.output,
            questions_path=args.questions,
            image_root=args.images,
            scenes_path=args.scenes,
            split=args.split,
        )

    print(
        f"Built {args.benchmark} manifest with {len(manifest)} examples at {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
