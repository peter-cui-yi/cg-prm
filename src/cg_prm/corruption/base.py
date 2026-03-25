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


def parse_gqa_grounding_ref(grounding_ref: str) -> tuple[str, dict[str, Any]]:
    """Parse the supported GQA grounding reference formats."""
    ref = grounding_ref.strip()
    if not ref:
        return "empty", {}
    if ref.startswith("object:"):
        object_id = ref.split(":", maxsplit=1)[1].strip()
        return "objects", {"object_ids": [object_id]}
    if ref.startswith("objects:"):
        raw_ids = ref.split(":", maxsplit=1)[1]
        object_ids = [value.strip() for value in raw_ids.split(",") if value.strip()]
        return "objects", {"object_ids": object_ids}
    if ref.startswith("relation:"):
        parts = ref.split(":", maxsplit=3)
        if len(parts) != 4:
            return "unknown", {"raw": ref}
        _, relation_name, source_id, target_id = parts
        if not relation_name.strip() or not source_id.strip() or not target_id.strip():
            return "unknown", {"raw": ref}
        return "relation", {
            "relation_name": relation_name.strip(),
            "source_id": source_id.strip(),
            "target_id": target_id.strip(),
        }
    return "unknown", {"raw": ref}


def format_gqa_object_ref(object_ids: list[str]) -> str:
    """Format a GQA object reference."""
    if len(object_ids) == 1:
        return f"object:{object_ids[0]}"
    return "objects:" + ",".join(object_ids)


def gqa_scene(example: NormalizedExample) -> dict[str, Any]:
    """Return scene-graph metadata for a GQA example."""
    raw_scene = example.metadata.get("scene_graph", {})
    return dict(raw_scene) if isinstance(raw_scene, Mapping) else {}


def gqa_objects(example: NormalizedExample) -> list[dict[str, Any]]:
    """Return GQA scene-graph objects."""
    scene = gqa_scene(example)
    objects = scene.get("objects", [])
    if not isinstance(objects, list):
        return []
    return [dict(obj) for obj in objects if isinstance(obj, Mapping)]


def choose_alternate_gqa_object_ids(
    example: NormalizedExample,
    *,
    current_ids: list[str],
    seed: int,
    salt: str,
) -> list[str] | None:
    """Choose alternate GQA object ids with matched cardinality."""
    objects = gqa_objects(example)
    available_ids = [str(obj.get("object_id") or "").strip() for obj in objects]
    candidates = [object_id for object_id in available_ids if object_id and object_id not in current_ids]
    if len(candidates) < len(current_ids):
        return None
    rng = seeded_random(seed, example.example_id, ",".join(current_ids), salt)
    rng.shuffle(candidates)
    return sorted(candidates[: len(current_ids)])


def gqa_object_attribute_summary(example: NormalizedExample, object_ids: list[str]) -> str:
    """Create a short attribute summary for a GQA object set."""
    objects = {
        str(obj.get("object_id") or "").strip(): obj
        for obj in gqa_objects(example)
        if str(obj.get("object_id") or "").strip()
    }
    summaries: list[str] = []
    for object_id in object_ids:
        obj = objects.get(object_id)
        if not obj:
            continue
        parts: list[str] = []
        attributes = obj.get("attributes", [])
        if isinstance(attributes, list):
            parts.extend(str(attribute).strip() for attribute in attributes[:2] if str(attribute).strip())
        name = str(obj.get("name") or "").strip()
        if name:
            parts.append(name)
        if parts:
            summaries.append(" ".join(parts))
    return ", ".join(summaries)


def visualwebbench_elements(example: NormalizedExample) -> list[dict[str, Any]]:
    """Return normalized VisualWebBench elements."""
    raw_elements = example.metadata.get("elements", [])
    if not isinstance(raw_elements, list):
        return []
    return [dict(element) for element in raw_elements if isinstance(element, Mapping)]


def lookup_visualwebbench_element(
    example: NormalizedExample,
    grounding_ref: str,
) -> dict[str, Any] | None:
    """Find a UI element by grounding ref."""
    if not grounding_ref:
        return None
    candidate_ids = [grounding_ref]
    if ":" in grounding_ref:
        candidate_ids.append(grounding_ref.split(":")[-1])
    for element in visualwebbench_elements(example):
        element_id = str(element.get("element_id") or "").strip()
        if element_id in candidate_ids:
            return element
    return None


def choose_alternate_visualwebbench_element(
    example: NormalizedExample,
    *,
    current_ref: str,
    seed: int,
    salt: str,
) -> dict[str, Any] | None:
    """Choose a different UI element for a VisualWebBench corruption."""
    current = lookup_visualwebbench_element(example, current_ref)
    current_norm = normalize_text(current.get("text")) if current else ""
    candidates = [
        element
        for element in visualwebbench_elements(example)
        if str(element.get("element_id") or "").strip()
        and normalize_text(element.get("text")) != current_norm
    ]
    if not candidates:
        return None
    rng = seeded_random(seed, example.example_id, current_ref, salt)
    return candidates[rng.randrange(len(candidates))]


def parse_visualwebbench_grounding_ref(grounding_ref: str) -> tuple[str, dict[str, Any]]:
    """Parse the supported VisualWebBench grounding reference formats."""
    ref = grounding_ref.strip()
    if not ref:
        return "empty", {}
    if ref.startswith("element:"):
        element_id = ref.split(":", maxsplit=1)[1].strip()
        return "elements", {"element_ids": [element_id]}
    if ref.startswith("elements:"):
        raw_ids = ref.split(":", maxsplit=1)[1]
        element_ids = [value.strip() for value in raw_ids.split(",") if value.strip()]
        return "elements", {"element_ids": element_ids}
    return "unknown", {"raw": ref}


def format_visualwebbench_element_ref(element_ids: list[str]) -> str:
    """Format a VisualWebBench element reference."""
    if len(element_ids) == 1:
        return f"element:{element_ids[0]}"
    return "elements:" + ",".join(element_ids)
