"""Independent secondary corruptor used for cross-corruptor evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from cg_prm.corruption.base import (
    CORRUPTION_FAMILIES,
    choose_alternate_docvqa_span,
    choose_alternate_gqa_object_ids,
    choose_alternate_visualwebbench_element,
    choose_step,
    finalize_corrupted_trace,
    format_gqa_object_ref,
    format_visualwebbench_element_ref,
    gqa_object_attribute_summary,
    mutate_inference_text,
    mutate_relation_text,
    parse_gqa_grounding_ref,
)
from cg_prm.data.schema import NormalizedExample, TraceRecord


def _cross_wrong_region(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="cross_wrong_region",
        require_grounding=True,
        reverse=True,
    )
    if target is None:
        return None
    if example.benchmark == "docvqa":
        alternate = choose_alternate_docvqa_span(
            example,
            current_ref=target.grounding_ref,
            seed=seed,
            salt="cross_wrong_region",
        )
        if alternate is None:
            return None
        mutated = replace(
            target,
            grounding_ref=f"ocr_span:{alternate['span_id']}",
            step_text=f"{target.step_text.rstrip('.')} I may be referring to a nearby field.",
        )
        extra = {"alternate_span_id": alternate["span_id"]}
    elif example.benchmark == "gqa":
        kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
        if kind != "objects":
            return None
        alternate_ids = choose_alternate_gqa_object_ids(
            example,
            current_ids=payload["object_ids"],
            seed=seed,
            salt="cross_wrong_region",
        )
        if alternate_ids is None:
            return None
        mutated = replace(
            target,
            grounding_ref=format_gqa_object_ref(alternate_ids),
            step_text=f"{target.step_text.rstrip('.')} I focus on another object set.",
        )
        extra = {"alternate_object_ids": alternate_ids}
    else:
        alternate = choose_alternate_visualwebbench_element(
            example,
            current_ref=target.grounding_ref,
            seed=seed,
            salt="cross_wrong_region",
        )
        if alternate is None:
            return None
        mutated = replace(
            target,
            grounding_ref=format_visualwebbench_element_ref([alternate["element_id"]]),
            step_text=f"{target.step_text.rstrip('.')} I may be looking at a different control.",
        )
        extra = {"alternate_element_id": alternate["element_id"]}
    return finalize_corrupted_trace(
        trace,
        family="wrong_region",
        generator_name="cross",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata=extra,
    )


def _cross_wrong_value(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(trace, seed=seed, family="cross_wrong_value", reverse=True)
    if target is None:
        return None
    if example.benchmark == "docvqa":
        alternate = choose_alternate_docvqa_span(
            example,
            current_ref=target.grounding_ref,
            seed=seed,
            salt="cross_wrong_value",
        )
        if alternate is None:
            return None
        wrong_text = str(alternate.get("text") or "").strip()
    elif example.benchmark == "gqa":
        kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
        if kind == "objects":
            alternate_ids = choose_alternate_gqa_object_ids(
                example,
                current_ids=payload["object_ids"],
                seed=seed,
                salt="cross_wrong_value",
            )
            if alternate_ids is None:
                return None
            wrong_text = gqa_object_attribute_summary(example, alternate_ids) or target.evidence_value
        else:
            wrong_text = f"not {target.evidence_value}".strip()
    else:
        alternate = choose_alternate_visualwebbench_element(
            example,
            current_ref=target.grounding_ref,
            seed=seed,
            salt="cross_wrong_value",
        )
        if alternate is None:
            return None
        wrong_text = str(alternate.get("text") or "").strip() or "different UI element"
    mutated = replace(
        target,
        evidence_value=wrong_text,
        step_text=f"{target.step_text.rstrip('.')} The evidence may read as {wrong_text}.",
    )
    return finalize_corrupted_trace(
        trace,
        family="wrong_value",
        generator_name="cross",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
    )


def _cross_wrong_relation(_example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(trace, seed=seed, family="cross_wrong_relation", reverse=True)
    if target is None:
        return None
    mutated = replace(
        target,
        step_text=mutate_relation_text(mutate_inference_text(target.step_text)),
    )
    return finalize_corrupted_trace(
        trace,
        family="wrong_relation",
        generator_name="cross",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
    )


def _cross_irrelevant(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(trace, seed=seed, family="cross_irrelevant", reverse=True)
    if target is None:
        return None
    if example.benchmark == "docvqa":
        alternate = choose_alternate_docvqa_span(
            example,
            current_ref=target.grounding_ref,
            seed=seed,
            salt="cross_irrelevant",
        )
        if alternate is None:
            return None
        mutated = replace(
            target,
            grounding_ref=f"ocr_span:{alternate['span_id']}",
            evidence_value=str(alternate.get("text") or "").strip(),
            step_text=f"{target.step_text.rstrip('.')} I also consider a nearby but less relevant field.",
        )
    elif example.benchmark == "gqa":
        kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
        if kind != "objects":
            return None
        alternate_ids = choose_alternate_gqa_object_ids(
            example,
            current_ids=payload["object_ids"],
            seed=seed,
            salt="cross_irrelevant",
        )
        if alternate_ids is None:
            return None
        mutated = replace(
            target,
            grounding_ref=format_gqa_object_ref(alternate_ids),
            evidence_value=gqa_object_attribute_summary(example, alternate_ids) or target.evidence_value,
            step_text=f"{target.step_text.rstrip('.')} I also inspect another object group.",
        )
    else:
        alternate = choose_alternate_visualwebbench_element(
            example,
            current_ref=target.grounding_ref,
            seed=seed,
            salt="cross_irrelevant",
        )
        if alternate is None:
            return None
        mutated = replace(
            target,
            grounding_ref=format_visualwebbench_element_ref([alternate["element_id"]]),
            evidence_value=str(alternate.get("text") or "").strip(),
            step_text=f"{target.step_text.rstrip('.')} I also consider another nearby interface element.",
        )
    return finalize_corrupted_trace(
        trace,
        family="irrelevant_evidence",
        generator_name="cross",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
    )


def _cross_wrong_intermediate(_example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(trace, seed=seed, family="cross_wrong_intermediate", reverse=True)
    if target is None:
        return None
    mutated = replace(target, step_text=mutate_inference_text(target.step_text))
    return finalize_corrupted_trace(
        trace,
        family="wrong_intermediate_evidence",
        generator_name="cross",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"answer_preserved": True},
    )


def generate_cross_corruptor_traces(
    example: NormalizedExample,
    trace: TraceRecord,
    *,
    seed: int = 0,
    families: Iterable[str] | None = None,
) -> list[TraceRecord]:
    """Generate one corruption per family with an independent mutation style."""
    requested_families = list(families or CORRUPTION_FAMILIES)
    generators = {
        "wrong_region": _cross_wrong_region,
        "wrong_value": _cross_wrong_value,
        "wrong_relation": _cross_wrong_relation,
        "irrelevant_evidence": _cross_irrelevant,
        "wrong_intermediate_evidence": _cross_wrong_intermediate,
    }
    traces: list[TraceRecord] = []
    for family in requested_families:
        if family not in generators:
            raise ValueError(f"Unsupported corruption family `{family}`.")
        trace_variant = generators[family](example, trace, seed)
        if trace_variant is not None:
            traces.append(trace_variant)
    return traces
