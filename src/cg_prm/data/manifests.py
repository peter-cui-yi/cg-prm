"""Utilities for working with normalized benchmark manifests."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cg_prm.data.schema import NormalizedExample, read_jsonl, write_jsonl


def load_manifest(path: str | Path) -> list[NormalizedExample]:
    """Load a JSONL manifest into validated `NormalizedExample` objects."""
    return [NormalizedExample.from_dict(record) for record in read_jsonl(path)]


def write_manifest(path: str | Path, examples: list[NormalizedExample]) -> None:
    """Write validated examples to a JSONL manifest."""
    write_jsonl(path, examples)


def index_examples_by_id(examples: list[NormalizedExample]) -> dict[str, NormalizedExample]:
    """Index normalized examples by example id."""
    lookup: dict[str, NormalizedExample] = {}
    for example in examples:
        if example.example_id in lookup:
            raise ValueError(f"Duplicate example id `{example.example_id}` in manifest.")
        lookup[example.example_id] = example
    return lookup


def group_examples_by_benchmark(
    examples: list[NormalizedExample],
) -> dict[str, list[NormalizedExample]]:
    """Group normalized examples by benchmark name."""
    grouped: dict[str, list[NormalizedExample]] = defaultdict(list)
    for example in examples:
        grouped[example.benchmark].append(example)
    return dict(grouped)
