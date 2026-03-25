"""Shared corruption helpers for counterfactual trace generation."""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Any, Iterable, Mapping

from cg_prm.data.schema import NormalizedExample, TraceRecord, TraceStep, normalize_text

CORRUPTION_FAMILIES = (
    "wrong_region",
    "wrong_value",
    "wrong_relation",
    "irrelevant_evidence",
    "wrong_intermediate_evidence",
)
WRONG_USE_TYPES = (
    "wrong_attribute_readout",
    "wrong_relation_composition",
    "wrong_inference_from_correct_facts",
)

RELATION_SWAPS = (
    ("left of", "right of"),
    ("right of", "left of"),
    ("before", "after"),
    ("after", "before"),
    ("above", "below"),
    ("below", "above"),
    ("front of", "behind"),
    ("behind", "front of"),
    ("header", "footer"),
    ("footer", "header"),
)


def seeded_random(seed: int, *salts: Any) -> random.Random:
    """Create a deterministic RNG from one base seed and optional salts."""
    material = "|".join(str(part) for part in (seed, *salts))
    return random.Random(material)


def candidate_steps(
    trace: TraceRecord,
    *,
    include_answer: bool = False,
    require_grounding: bool = False,
    preferred_types: Iterable[str] | None = None,
) -> list[TraceStep]:
    """Select candidate steps for corruption."""
    preferred = set(preferred_types or [])
    steps = []
    for step in trace.steps:
        if not include_answer and step.step_type == "answer":
            continue
        if require_grounding and not step.grounding_ref:
            continue
        steps.append(step)
    if preferred:
        preferred_steps = [step for step in steps if step.step_type in preferred]
        if preferred_steps:
            return preferred_steps
    return steps


def choose_step(
    trace: TraceRecord,
    *,
    seed: int,
    family: str,
    include_answer: bool = False,
    require_grounding: bool = False,
    preferred_types: Iterable[str] | None = None,
    reverse: bool = False,
) -> TraceStep | None:
    """Choose one target step deterministically."""
    steps = candidate_steps(
        trace,
        include_answer=include_answer,
        require_grounding=require_grounding,
        preferred_types=preferred_types,
    )
    if not steps:
        return None
    rng = seeded_random(seed, trace.trace_id, family, "step")
    ordered = list(reversed(steps)) if reverse else list(steps)
    return ordered[rng.randrange(len(ordered))]


def replace_words(text: str, replacements: Iterable[tuple[str, str]]) -> str:
    """Apply the first matching phrase replacement with word boundaries."""
    updated = text
    for source, target in replacements:
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
        if pattern.search(updated):
            return pattern.sub(target, updated, count=1)
    return updated


def mutate_relation_text(text: str) -> str:
    """Change a relation phrase while preserving local fluency."""
    replaced = replace_words(text, RELATION_SWAPS)
    if replaced != text:
        return replaced
    return f"{text.rstrip('.')} using a different relation."


def mutate_inference_text(text: str) -> str:
    """Make a reasoning step read like a flawed inference."""
    lowered = text.lower()
    if "therefore" in lowered or "thus" in lowered or "so" in lowered:
        return replace_words(
            text,
            (
                ("therefore", "however"),
                ("thus", "instead"),
                ("so", "still"),
            ),
        )
    return f"{text.rstrip('.')} This suggests a different conclusion."


def merge_metadata(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Merge metadata mappings without mutating the original input."""
    merged = dict(base)
    for key, value in extra.items():
        merged[key] = value
    return merged


def finalize_corrupted_trace(
    trace: TraceRecord,
    *,
    family: str,
    generator_name: str,
    target_step_id: int,
    mutated_step: TraceStep,
    preserve_answer: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> TraceRecord:
    """Return a new trace with one step marked as corrupted."""
    steps: list[TraceStep] = []
    for step in trace.steps:
        if step.step_id != target_step_id:
            steps.append(step)
            continue
        corruption_metadata = {
            "family": family,
            "generator_name": generator_name,
            "source_trace_id": trace.trace_id,
            "target_step_id": target_step_id,
            "preserve_answer": preserve_answer,
        }
        if extra_metadata:
            corruption_metadata.update(dict(extra_metadata))
        merged_step_metadata = merge_metadata(
            mutated_step.metadata,
            {"corruption": corruption_metadata},
        )
        steps.append(
            replace(
                mutated_step,
                label=0,
                error_type=family,
                metadata=merged_step_metadata,
            )
        )

    trace_metadata = merge_metadata(
        trace.metadata,
        {
            "corruption": {
                "family": family,
                "generator_name": generator_name,
                "source_trace_id": trace.trace_id,
                "target_step_id": target_step_id,
                "preserve_answer": preserve_answer,
                **(dict(extra_metadata) if extra_metadata else {}),
            }
        },
    )
    return replace(
        trace,
        trace_id=f"{trace.trace_id}__{generator_name}__{family}__s{target_step_id}",
        steps=steps,
        metadata=trace_metadata,
    )


def docvqa_spans(example: NormalizedExample) -> list[dict[str, Any]]:
    """Return OCR spans for a DocVQA example."""
    raw_spans = example.metadata.get("ocr_spans", [])
    if not isinstance(raw_spans, list):
        return []
    return [dict(span) for span in raw_spans if isinstance(span, Mapping)]


def lookup_docvqa_span(example: NormalizedExample, grounding_ref: str) -> dict[str, Any] | None:
    """Find an OCR span by grounding ref."""
    if not grounding_ref:
        return None
    candidate_ids = [grounding_ref]
    if ":" in grounding_ref:
        candidate_ids.append(grounding_ref.split(":")[-1])
    for span in docvqa_spans(example):
        span_id = str(span.get("span_id") or "").strip()
        if span_id in candidate_ids:
            return span
    return None


def choose_alternate_docvqa_span(
    example: NormalizedExample,
    *,
    current_ref: str,
    seed: int,
    salt: str,
) -> dict[str, Any] | None:
    """Choose a different OCR span for a DocVQA corruption."""
    current = lookup_docvqa_span(example, current_ref)
    current_norm = normalize_text(current.get("text")) if current else ""
    candidates = [
        span
        for span in docvqa_spans(example)
        if str(span.get("span_id") or "").strip()
        and normalize_text(span.get("text")) != current_norm
    ]
    if not candidates:
        return None
    rng = seeded_random(seed, example.example_id, current_ref, salt)
    return candidates[rng.randrange(len(candidates))]


def parse_clevr_grounding_ref(grounding_ref: str) -> tuple[str, dict[str, Any]]:
    """Parse the supported CLEVR grounding reference formats."""
    ref = grounding_ref.strip()
    if not ref:
        return "empty", {}
    if ref.startswith("object:"):
        object_id = int(ref.split(":", maxsplit=1)[1])
        return "objects", {"object_ids": [object_id]}
    if ref.startswith("objects:"):
        raw_ids = ref.split(":", maxsplit=1)[1]
        object_ids = [int(value.strip()) for value in raw_ids.split(",") if value.strip()]
        return "objects", {"object_ids": object_ids}
    if ref.startswith("relation:"):
        _, relation_name, source_id, target_id = ref.split(":", maxsplit=3)
        return "relation", {
            "relation_name": relation_name,
            "source_id": int(source_id),
            "target_id": int(target_id),
        }
    return "unknown", {"raw": ref}


def format_clevr_object_ref(object_ids: list[int]) -> str:
    """Format a CLEVR object reference."""
    if len(object_ids) == 1:
        return f"object:{object_ids[0]}"
    return "objects:" + ",".join(str(object_id) for object_id in object_ids)


def clevr_scene(example: NormalizedExample) -> dict[str, Any]:
    """Return scene metadata for a CLEVR example."""
    raw_scene = example.metadata.get("scene", {})
    return dict(raw_scene) if isinstance(raw_scene, Mapping) else {}


def clevr_objects(example: NormalizedExample) -> list[dict[str, Any]]:
    """Return CLEVR scene objects."""
    scene = clevr_scene(example)
    objects = scene.get("objects", [])
    if not isinstance(objects, list):
        return []
    return [dict(obj) for obj in objects if isinstance(obj, Mapping)]


def choose_alternate_clevr_object_ids(
    example: NormalizedExample,
    *,
    current_ids: list[int],
    seed: int,
    salt: str,
) -> list[int] | None:
    """Choose alternate CLEVR object ids with matched cardinality."""
    objects = clevr_objects(example)
    available_ids = [int(obj["object_id"]) for obj in objects if "object_id" in obj]
    candidates = [object_id for object_id in available_ids if object_id not in current_ids]
    if len(candidates) < len(current_ids):
        return None
    rng = seeded_random(seed, example.example_id, ",".join(map(str, current_ids)), salt)
    rng.shuffle(candidates)
    return sorted(candidates[: len(current_ids)])


def object_attribute_summary(example: NormalizedExample, object_ids: list[int]) -> str:
    """Create a short attribute summary for a CLEVR object set."""
    objects = {int(obj["object_id"]): obj for obj in clevr_objects(example) if "object_id" in obj}
    summaries: list[str] = []
    for object_id in object_ids:
        obj = objects.get(object_id)
        if not obj:
            continue
        parts = [str(obj.get(key)).strip() for key in ("color", "shape") if obj.get(key)]
        if parts:
            summaries.append(" ".join(parts))
    return ", ".join(summaries)
