"""Unified benchmark-aware validation for clean trace verification."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from cg_prm.data.schema import NormalizedExample, TraceRecord, TraceStep, normalize_text

GROUNDING_REQUIRED_STEP_TYPES = frozenset(
    {"locate", "read", "extract", "identify", "relate", "count", "compute", "reason", "verify"}
)
EVIDENCE_REQUIRED_STEP_TYPES = frozenset({"read", "extract", "count", "compute"})
OBJECT_ATTRIBUTE_KEYS = ("color", "shape", "size", "material")


@dataclass(slots=True)
class ValidationIssue:
    """One concrete validation issue on a step or full trace."""

    code: str
    message: str
    step_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "step_id": self.step_id,
        }


@dataclass(slots=True)
class StepValidationResult:
    """Validation result for one trace step."""

    step_id: int
    passed: bool
    label: int
    error_type: str
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "passed": self.passed,
            "label": self.label,
            "error_type": self.error_type,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TraceValidationResult:
    """Validation result for a full trace."""

    example_id: str
    benchmark: str
    passed: bool
    answer_correct: bool | None
    step_results: list[StepValidationResult]
    issues: list[ValidationIssue] = field(default_factory=list)
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "benchmark": self.benchmark,
            "passed": self.passed,
            "answer_correct": self.answer_correct,
            "step_results": [result.to_dict() for result in self.step_results],
            "issues": [issue.to_dict() for issue in self.issues],
            "rejection_reason": self.rejection_reason,
        }


def _make_issue(code: str, message: str, step_id: int | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, step_id=step_id)


def _error_type_from_issues(issues: list[ValidationIssue]) -> str:
    return issues[0].code if issues else "none"


def _normalize_answer(value: str | None) -> str:
    return normalize_text(value or "")


def _validate_docvqa_step(
    step: TraceStep,
    *,
    spans_by_id: Mapping[str, Mapping[str, Any]],
) -> StepValidationResult:
    issues: list[ValidationIssue] = []
    metadata: dict[str, Any] = {}
    matched_span: Mapping[str, Any] | None = None

    if step.step_type in GROUNDING_REQUIRED_STEP_TYPES and not step.grounding_ref:
        issues.append(
            _make_issue(
                "missing_grounding_ref",
                "Grounding-critical DocVQA step is missing `grounding_ref`.",
                step.step_id,
            )
        )

    if step.grounding_ref:
        span_id = step.grounding_ref.split(":")[-1]
        matched_span = spans_by_id.get(span_id) or spans_by_id.get(step.grounding_ref)
        if matched_span is None:
            issues.append(
                _make_issue(
                    "unknown_grounding_ref",
                    f"DocVQA OCR span `{step.grounding_ref}` is not present in metadata.",
                    step.step_id,
                )
            )
        else:
            metadata["matched_span_id"] = matched_span.get("span_id")
            metadata["matched_span_text"] = matched_span.get("text")

    if step.step_type in EVIDENCE_REQUIRED_STEP_TYPES and not step.evidence_value:
        issues.append(
            _make_issue(
                "missing_evidence_value",
                "Evidence-bearing DocVQA step is missing `evidence_value`.",
                step.step_id,
            )
        )

    if matched_span is not None and step.evidence_value:
        span_text = _normalize_answer(str(matched_span.get("normalized_text") or matched_span.get("text")))
        evidence_text = _normalize_answer(step.evidence_value)
        if evidence_text and evidence_text not in span_text and span_text not in evidence_text:
            issues.append(
                _make_issue(
                    "evidence_mismatch",
                    "DocVQA evidence value does not align with the grounded OCR span.",
                    step.step_id,
                )
            )

    passed = not issues
    return StepValidationResult(
        step_id=step.step_id,
        passed=passed,
        label=int(passed),
        error_type=_error_type_from_issues(issues),
        issues=issues,
        metadata=metadata,
    )


def _object_lookup(scene: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    lookup: dict[int, Mapping[str, Any]] = {}
    for item in scene.get("objects", []):
        if not isinstance(item, Mapping):
            continue
        object_id = item.get("object_id")
        if isinstance(object_id, int):
            lookup[object_id] = item
    return lookup


def _parse_clevr_grounding_ref(grounding_ref: str) -> tuple[str, dict[str, Any]]:
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


def _validate_clevr_step(
    step: TraceStep,
    *,
    scene: Mapping[str, Any],
) -> StepValidationResult:
    issues: list[ValidationIssue] = []
    metadata: dict[str, Any] = {}
    objects = _object_lookup(scene)
    relationships = scene.get("relationships") if isinstance(scene.get("relationships"), Mapping) else {}

    if step.step_type in GROUNDING_REQUIRED_STEP_TYPES and not step.grounding_ref:
        issues.append(
            _make_issue(
                "missing_grounding_ref",
                "Grounding-critical CLEVR step is missing `grounding_ref`.",
                step.step_id,
            )
        )

    parsed_kind, parsed_payload = _parse_clevr_grounding_ref(step.grounding_ref)
    if step.grounding_ref and parsed_kind == "unknown":
        issues.append(
            _make_issue(
                "malformed_grounding_ref",
                f"CLEVR grounding reference `{step.grounding_ref}` does not match a supported format.",
                step.step_id,
            )
        )

    if parsed_kind == "objects":
        object_ids = parsed_payload["object_ids"]
        metadata["object_ids"] = object_ids
        missing = [object_id for object_id in object_ids if object_id not in objects]
        if missing:
            issues.append(
                _make_issue(
                    "unknown_grounding_ref",
                    f"CLEVR object ids {missing} are not present in the scene metadata.",
                    step.step_id,
                )
            )
        elif step.step_type == "count" and step.evidence_value:
            try:
                if int(step.evidence_value) != len(object_ids):
                    issues.append(
                        _make_issue(
                            "evidence_mismatch",
                            "CLEVR count step evidence does not match the grounded object set size.",
                            step.step_id,
                        )
                    )
            except ValueError:
                issues.append(
                    _make_issue(
                        "evidence_mismatch",
                        "CLEVR count step evidence is not an integer.",
                        step.step_id,
                    )
                )
        elif step.evidence_value and len(object_ids) == 1:
            object_payload = objects[object_ids[0]]
            attribute_values = {
                normalize_text(object_payload.get(attribute))
                for attribute in OBJECT_ATTRIBUTE_KEYS
                if object_payload.get(attribute) is not None
            }
            evidence_value = normalize_text(step.evidence_value)
            if (
                evidence_value
                and attribute_values
                and evidence_value not in attribute_values
                and evidence_value != normalize_text(object_ids[0])
            ):
                metadata["attribute_values"] = sorted(attribute_values)
                issues.append(
                    _make_issue(
                        "evidence_mismatch",
                        "CLEVR evidence value does not match the grounded object's attributes.",
                        step.step_id,
                    )
                )

    if parsed_kind == "relation":
        relation_name = parsed_payload["relation_name"]
        source_id = parsed_payload["source_id"]
        target_id = parsed_payload["target_id"]
        metadata["relation_name"] = relation_name
        metadata["source_id"] = source_id
        metadata["target_id"] = target_id
        relation_lists = relationships.get(relation_name)
        if not isinstance(relation_lists, list):
            issues.append(
                _make_issue(
                    "unknown_grounding_ref",
                    f"CLEVR relation `{relation_name}` is not available in scene metadata.",
                    step.step_id,
                )
            )
        elif source_id >= len(relation_lists) or source_id < 0:
            issues.append(
                _make_issue(
                    "unknown_grounding_ref",
                    f"CLEVR source object id `{source_id}` is out of range for relation `{relation_name}`.",
                    step.step_id,
                )
            )
        else:
            valid_targets = relation_lists[source_id]
            if target_id not in valid_targets:
                issues.append(
                    _make_issue(
                        "evidence_mismatch",
                        "CLEVR relation step is inconsistent with scene metadata.",
                        step.step_id,
                    )
                )

    if step.step_type in EVIDENCE_REQUIRED_STEP_TYPES and not step.evidence_value:
        issues.append(
            _make_issue(
                "missing_evidence_value",
                "Evidence-bearing CLEVR step is missing `evidence_value`.",
                step.step_id,
            )
        )

    passed = not issues
    return StepValidationResult(
        step_id=step.step_id,
        passed=passed,
        label=int(passed),
        error_type=_error_type_from_issues(issues),
        issues=issues,
        metadata=metadata,
    )


def validate_trace(example: NormalizedExample, trace: TraceRecord) -> TraceValidationResult:
    """Validate a trace against benchmark-specific evidence sources."""
    issues: list[ValidationIssue] = []
    if example.example_id != trace.example_id:
        issues.append(
            _make_issue(
                "validation_error",
                f"Trace example id `{trace.example_id}` does not match manifest id `{example.example_id}`.",
            )
        )
    if example.benchmark != trace.benchmark:
        issues.append(
            _make_issue(
                "validation_error",
                f"Trace benchmark `{trace.benchmark}` does not match manifest benchmark `{example.benchmark}`.",
            )
        )

    step_results: list[StepValidationResult] = []
    if trace.benchmark == "docvqa":
        raw_spans = example.metadata.get("ocr_spans", [])
        spans_by_id = {}
        if isinstance(raw_spans, list):
            for span in raw_spans:
                if isinstance(span, Mapping):
                    span_id = str(span.get("span_id") or "").strip()
                    if span_id:
                        spans_by_id[span_id] = span
        for step in trace.steps:
            step_results.append(_validate_docvqa_step(step, spans_by_id=spans_by_id))
    elif trace.benchmark == "clevr":
        scene = example.metadata.get("scene", {})
        if not isinstance(scene, Mapping):
            scene = {}
        for step in trace.steps:
            step_results.append(_validate_clevr_step(step, scene=scene))
    else:
        raise ValueError(f"Unsupported benchmark `{trace.benchmark}`.")

    answer_correct: bool | None = None
    if trace.predicted_answer is not None:
        answer_correct = _normalize_answer(trace.predicted_answer) == _normalize_answer(example.answer)
        if not answer_correct:
            issues.append(
                _make_issue(
                    "wrong_answer",
                    "Predicted answer does not match the normalized gold answer.",
                )
            )

    step_pass = all(result.passed for result in step_results)
    passed = step_pass and (answer_correct is not False) and not any(
        issue.code == "validation_error" for issue in issues
    )
    rejection_reason = None
    if not passed:
        if any(issue.code == "validation_error" for issue in issues):
            rejection_reason = "trace_metadata_mismatch"
        elif answer_correct is False:
            rejection_reason = "wrong_answer"
        else:
            first_failed = next((result for result in step_results if not result.passed), None)
            rejection_reason = first_failed.error_type if first_failed else "validation_error"

    return TraceValidationResult(
        example_id=example.example_id,
        benchmark=example.benchmark,
        passed=passed,
        answer_correct=answer_correct,
        step_results=step_results,
        issues=issues,
        rejection_reason=rejection_reason,
    )


def annotate_trace_with_validation(
    trace: TraceRecord,
    validation_result: TraceValidationResult,
) -> TraceRecord:
    """Return a new trace with validation labels and metadata attached."""
    step_lookup = {result.step_id: result for result in validation_result.step_results}
    annotated_steps: list[TraceStep] = []
    for step in trace.steps:
        result = step_lookup[step.step_id]
        metadata = dict(step.metadata)
        metadata["validation"] = result.to_dict()
        error_type = "none" if result.passed else result.error_type
        annotated_steps.append(
            replace(
                step,
                label=result.label,
                error_type=error_type,
                metadata=metadata,
            )
        )

    metadata = dict(trace.metadata)
    metadata["validation"] = validation_result.to_dict()
    return replace(trace, steps=annotated_steps, metadata=metadata)
