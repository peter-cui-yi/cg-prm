"""Main corruption families for CG-PRM counterfactual supervision."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from cg_prm.corruption.base import (
    CORRUPTION_FAMILIES,
    candidate_steps,
    choose_alternate_docvqa_span,
    choose_alternate_gqa_object_ids,
    choose_alternate_visualwebbench_element,
    choose_step,
    finalize_corrupted_trace,
    format_gqa_object_ref,
    format_visualwebbench_element_ref,
    gqa_object_attribute_summary,
    gqa_scene,
    mutate_inference_text,
    mutate_relation_text,
    parse_gqa_grounding_ref,
)
from cg_prm.data.schema import NormalizedExample, TraceRecord


def _docvqa_wrong_region(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_region",
        require_grounding=True,
        preferred_types=("locate", "read", "extract"),
    )
    if target is None:
        return None
    alternate = choose_alternate_docvqa_span(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="wrong_region",
    )
    if alternate is None:
        return None
    mutated = replace(target, grounding_ref=f"ocr_span:{alternate['span_id']}")
    return finalize_corrupted_trace(
        trace,
        family="wrong_region",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_span_id": alternate["span_id"]},
    )


def _docvqa_wrong_value(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_value",
        preferred_types=("read", "extract", "compute"),
    )
    if target is None or not target.grounding_ref:
        return None
    alternate = choose_alternate_docvqa_span(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="wrong_value",
    )
    if alternate is None:
        return None
    mutated = replace(target, evidence_value=str(alternate.get("text") or "").strip())
    return finalize_corrupted_trace(
        trace,
        family="wrong_value",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_span_id": alternate["span_id"]},
    )


def _docvqa_wrong_relation(_example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_relation",
        preferred_types=("reason", "derive", "compute", "locate"),
    )
    if target is None:
        return None
    mutated = replace(target, step_text=mutate_relation_text(target.step_text))
    return finalize_corrupted_trace(
        trace,
        family="wrong_relation",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"wrong_use_type": "wrong_relation_composition"},
    )


def _docvqa_irrelevant_evidence(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="irrelevant_evidence",
        preferred_types=("read", "extract", "locate"),
    )
    if target is None:
        return None
    alternate = choose_alternate_docvqa_span(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="irrelevant_evidence",
    )
    if alternate is None:
        return None
    mutated = replace(
        target,
        grounding_ref=f"ocr_span:{alternate['span_id']}",
        evidence_value=str(alternate.get("text") or "").strip(),
    )
    return finalize_corrupted_trace(
        trace,
        family="irrelevant_evidence",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_span_id": alternate["span_id"]},
    )


def _docvqa_wrong_intermediate(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_intermediate_evidence",
        preferred_types=("read", "extract", "reason", "derive"),
    )
    if target is None:
        return None
    alternate = choose_alternate_docvqa_span(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="wrong_intermediate",
    )
    wrong_value = str(alternate.get("text") or "").strip() if alternate is not None else target.evidence_value
    mutated = replace(
        target,
        evidence_value=wrong_value,
        step_text=mutate_inference_text(target.step_text),
    )
    return finalize_corrupted_trace(
        trace,
        family="wrong_intermediate_evidence",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"answer_preserved": True},
    )


def _gqa_wrong_region(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_region",
        require_grounding=True,
        preferred_types=("identify", "count", "relate", "extract"),
    )
    if target is None:
        return None
    kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
    if kind != "objects":
        return None
    alternate_ids = choose_alternate_gqa_object_ids(
        example,
        current_ids=payload["object_ids"],
        seed=seed,
        salt="wrong_region",
    )
    if alternate_ids is None:
        return None
    mutated = replace(target, grounding_ref=format_gqa_object_ref(alternate_ids))
    return finalize_corrupted_trace(
        trace,
        family="wrong_region",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_object_ids": alternate_ids},
    )


def _gqa_wrong_value(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_value",
        preferred_types=("identify", "count", "extract", "compute"),
    )
    if target is None:
        return None
    kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
    if kind == "objects":
        alternate_ids = choose_alternate_gqa_object_ids(
            example,
            current_ids=payload["object_ids"],
            seed=seed,
            salt="wrong_value",
        )
        if alternate_ids is None:
            return None
        alternate_summary = gqa_object_attribute_summary(example, alternate_ids) or str(len(alternate_ids))
        mutated = replace(target, evidence_value=alternate_summary)
    elif kind == "relation":
        mutated = replace(target, step_text=mutate_relation_text(target.step_text))
    else:
        mutated = replace(target, evidence_value=f"not {target.evidence_value}".strip())
    return finalize_corrupted_trace(
        trace,
        family="wrong_value",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
    )


def _gqa_wrong_relation(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_relation",
        require_grounding=True,
        preferred_types=("relate", "reason", "derive"),
    )
    if target is None:
        return None
    kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
    mutated = None
    if kind == "relation":
        relation_name = str(payload["relation_name"])
        scene = gqa_scene(example)
        relationships = {}
        for item in scene.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or "").strip()
            if object_id:
                relationships[object_id] = item.get("relations", [])
        swapped_name = mutate_relation_text(relation_name).split()[0]
        source_relations = relationships.get(payload["source_id"], [])
        for relation in source_relations:
            if not isinstance(relation, dict):
                continue
            if str(relation.get("name") or "").strip() == swapped_name:
                mutated = replace(
                    target,
                    grounding_ref=(
                        f"relation:{swapped_name}:{payload['source_id']}:{payload['target_id']}"
                    ),
                    step_text=mutate_relation_text(target.step_text),
                )
                break
    if mutated is None:
        mutated = replace(target, step_text=mutate_relation_text(target.step_text))
    return finalize_corrupted_trace(
        trace,
        family="wrong_relation",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"wrong_use_type": "wrong_relation_composition"},
    )


def _gqa_irrelevant_evidence(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="irrelevant_evidence",
        preferred_types=("identify", "extract", "count"),
    )
    if target is None:
        return None
    kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
    if kind != "objects":
        return None
    alternate_ids = choose_alternate_gqa_object_ids(
        example,
        current_ids=payload["object_ids"],
        seed=seed,
        salt="irrelevant_evidence",
    )
    if alternate_ids is None:
        return None
    mutated = replace(
        target,
        grounding_ref=format_gqa_object_ref(alternate_ids),
        evidence_value=gqa_object_attribute_summary(example, alternate_ids) or target.evidence_value,
    )
    return finalize_corrupted_trace(
        trace,
        family="irrelevant_evidence",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_object_ids": alternate_ids},
    )


def _gqa_wrong_intermediate(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_intermediate_evidence",
        preferred_types=("identify", "count", "reason", "derive"),
    )
    if target is None:
        return None
    kind, payload = parse_gqa_grounding_ref(target.grounding_ref)
    if kind == "objects":
        wrong_value = (
            str(max(0, len(payload["object_ids"]) - 1))
            if target.step_type == "count"
            else "different object set"
        )
        mutated = replace(
            target,
            evidence_value=wrong_value,
            step_text=mutate_inference_text(target.step_text),
        )
    else:
        mutated = replace(target, step_text=mutate_inference_text(target.step_text))
    return finalize_corrupted_trace(
        trace,
        family="wrong_intermediate_evidence",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"answer_preserved": True},
    )


def _visualwebbench_wrong_region(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_region",
        require_grounding=True,
        preferred_types=("locate", "read", "extract"),
    )
    if target is None:
        return None
    alternate = choose_alternate_visualwebbench_element(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="wrong_region",
    )
    if alternate is None:
        return None
    mutated = replace(target, grounding_ref=format_visualwebbench_element_ref([alternate["element_id"]]))
    return finalize_corrupted_trace(
        trace,
        family="wrong_region",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_element_id": alternate["element_id"]},
    )


def _visualwebbench_wrong_value(example: NormalizedExample, trace: TraceRecord, seed: int) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_value",
        preferred_types=("read", "extract", "compute"),
    )
    if target is None or not target.grounding_ref:
        return None
    alternate = choose_alternate_visualwebbench_element(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="wrong_value",
    )
    if alternate is None:
        return None
    mutated = replace(target, evidence_value=str(alternate.get("text") or "").strip() or "different element")
    return finalize_corrupted_trace(
        trace,
        family="wrong_value",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_element_id": alternate["element_id"]},
    )


def _visualwebbench_wrong_relation(
    _example: NormalizedExample,
    trace: TraceRecord,
    seed: int,
) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_relation",
        preferred_types=("reason", "derive", "compute", "locate"),
    )
    if target is None:
        return None
    mutated = replace(target, step_text=mutate_relation_text(target.step_text))
    return finalize_corrupted_trace(
        trace,
        family="wrong_relation",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"wrong_use_type": "wrong_relation_composition"},
    )


def _visualwebbench_irrelevant_evidence(
    example: NormalizedExample,
    trace: TraceRecord,
    seed: int,
) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="irrelevant_evidence",
        preferred_types=("read", "extract", "locate"),
    )
    if target is None:
        return None
    alternate = choose_alternate_visualwebbench_element(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="irrelevant_evidence",
    )
    if alternate is None:
        return None
    mutated = replace(
        target,
        grounding_ref=format_visualwebbench_element_ref([alternate["element_id"]]),
        evidence_value=str(alternate.get("text") or "").strip(),
    )
    return finalize_corrupted_trace(
        trace,
        family="irrelevant_evidence",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"alternate_element_id": alternate["element_id"]},
    )


def _visualwebbench_wrong_intermediate(
    example: NormalizedExample,
    trace: TraceRecord,
    seed: int,
) -> TraceRecord | None:
    target = choose_step(
        trace,
        seed=seed,
        family="wrong_intermediate_evidence",
        preferred_types=("read", "extract", "reason", "derive"),
    )
    if target is None:
        return None
    alternate = choose_alternate_visualwebbench_element(
        example,
        current_ref=target.grounding_ref,
        seed=seed,
        salt="wrong_intermediate",
    )
    wrong_value = (
        str(alternate.get("text") or "").strip()
        if alternate is not None
        else target.evidence_value or "different UI cue"
    )
    mutated = replace(
        target,
        evidence_value=wrong_value,
        step_text=mutate_inference_text(target.step_text),
    )
    return finalize_corrupted_trace(
        trace,
        family="wrong_intermediate_evidence",
        generator_name="main",
        target_step_id=target.step_id,
        mutated_step=mutated,
        preserve_answer=True,
        extra_metadata={"answer_preserved": True},
    )


MAIN_GENERATORS = {
    "docvqa": {
        "wrong_region": _docvqa_wrong_region,
        "wrong_value": _docvqa_wrong_value,
        "wrong_relation": _docvqa_wrong_relation,
        "irrelevant_evidence": _docvqa_irrelevant_evidence,
        "wrong_intermediate_evidence": _docvqa_wrong_intermediate,
    },
    "gqa": {
        "wrong_region": _gqa_wrong_region,
        "wrong_value": _gqa_wrong_value,
        "wrong_relation": _gqa_wrong_relation,
        "irrelevant_evidence": _gqa_irrelevant_evidence,
        "wrong_intermediate_evidence": _gqa_wrong_intermediate,
    },
    "visualwebbench": {
        "wrong_region": _visualwebbench_wrong_region,
        "wrong_value": _visualwebbench_wrong_value,
        "wrong_relation": _visualwebbench_wrong_relation,
        "irrelevant_evidence": _visualwebbench_irrelevant_evidence,
        "wrong_intermediate_evidence": _visualwebbench_wrong_intermediate,
    },
}


def generate_corrupted_traces(
    example: NormalizedExample,
    trace: TraceRecord,
    *,
    seed: int = 0,
    families: Iterable[str] | None = None,
) -> list[TraceRecord]:
    """Generate one corruption per family when possible."""
    requested_families = list(families or CORRUPTION_FAMILIES)
    invalid = [family for family in requested_families if family not in CORRUPTION_FAMILIES]
    if invalid:
        raise ValueError(f"Unsupported corruption families: {invalid}")
    generators = MAIN_GENERATORS[example.benchmark]
    corrupted: list[TraceRecord] = []
    for family in requested_families:
        trace_variant = generators[family](example, trace, seed)
        if trace_variant is not None:
            corrupted.append(trace_variant)
    return corrupted


def generate_wrong_use_traces(
    example: NormalizedExample,
    trace: TraceRecord,
    *,
    seed: int = 0,
) -> list[TraceRecord]:
    """Generate the three correct-evidence but wrong-use subtypes when possible."""
    traces: list[TraceRecord] = []

    attribute_generators = {
        "docvqa": _docvqa_wrong_value,
        "gqa": _gqa_wrong_value,
        "visualwebbench": _visualwebbench_wrong_value,
    }
    relation_generators = {
        "docvqa": _docvqa_wrong_relation,
        "gqa": _gqa_wrong_relation,
        "visualwebbench": _visualwebbench_wrong_relation,
    }

    attribute_trace = attribute_generators[example.benchmark](example, trace, seed + 11)
    if attribute_trace is not None:
        metadata = dict(attribute_trace.metadata.get("corruption", {}))
        metadata["wrong_use_type"] = "wrong_attribute_readout"
        attribute_trace = replace(
            attribute_trace,
            metadata={**attribute_trace.metadata, "corruption": metadata},
        )
        traces.append(attribute_trace)

    relation_trace = relation_generators[example.benchmark](example, trace, seed + 23)
    if relation_trace is not None:
        metadata = dict(relation_trace.metadata.get("corruption", {}))
        metadata["wrong_use_type"] = "wrong_relation_composition"
        relation_trace = replace(
            relation_trace,
            metadata={**relation_trace.metadata, "corruption": metadata},
        )
        traces.append(relation_trace)

    inference_target = choose_step(
        trace,
        seed=seed + 37,
        family="wrong_use_inference",
        preferred_types=("reason", "derive", "compute"),
    )
    if inference_target is None:
        inference_candidates = candidate_steps(trace, preferred_types=("reason", "derive", "compute"))
        inference_target = inference_candidates[0] if inference_candidates else None
    if inference_target is not None:
        mutated = replace(inference_target, step_text=mutate_inference_text(inference_target.step_text))
        traces.append(
            finalize_corrupted_trace(
                trace,
                family="wrong_intermediate_evidence",
                generator_name="main",
                target_step_id=inference_target.step_id,
                mutated_step=mutated,
                preserve_answer=True,
                extra_metadata={"wrong_use_type": "wrong_inference_from_correct_facts"},
            )
        )

    return traces
