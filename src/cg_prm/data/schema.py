"""Strict shared schema definitions for CG-PRM manifests and traces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

ALLOWED_BENCHMARKS = frozenset({"docvqa", "clevr"})
ALLOWED_TRACE_MODES = frozenset({"canonical", "light", "free"})
ALLOWED_STEP_TYPES = frozenset(
    {
        "locate",
        "read",
        "extract",
        "identify",
        "relate",
        "count",
        "compute",
        "derive",
        "answer",
        "reason",
        "verify",
        "free",
        "summary",
    }
)
ALLOWED_ERROR_TYPES = frozenset(
    {
        "none",
        "wrong_region",
        "wrong_value",
        "wrong_relation",
        "irrelevant_evidence",
        "wrong_intermediate_evidence",
        "missing_grounding_ref",
        "unknown_grounding_ref",
        "malformed_grounding_ref",
        "missing_evidence_value",
        "evidence_mismatch",
        "wrong_answer",
        "malformed_step",
        "validation_error",
    }
)


class SchemaValidationError(ValueError):
    """Raised when a manifest or trace does not satisfy the shared schema."""


def normalize_text(value: Any) -> str:
    """Normalize free-form text for tolerant matching across datasets."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[\W_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise SchemaValidationError(
                    f"JSONL line {line_number} of {path} is not an object."
                )
            records.append(payload)
    return records


def write_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any] | "NormalizedExample" | "TraceRecord"],
) -> None:
    """Write JSON serializable mappings or schema objects to JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "to_dict"):
                payload = record.to_dict()  # type: ignore[assignment]
            else:
                payload = dict(record)
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def iter_jsonl_objects(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL objects one by one."""
    for record in read_jsonl(path):
        yield record


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"`{field_name}` must be a mapping.")
    return dict(value)


def _require_non_empty_text(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise SchemaValidationError(f"`{field_name}` must be a non-empty string.")
    return text


def _coerce_label(value: Any) -> int:
    if value not in (0, 1):
        raise SchemaValidationError("`label` must be 0 or 1.")
    return int(value)


@dataclass(slots=True)
class NormalizedExample:
    """Normalized benchmark example used by all downstream modules."""

    example_id: str
    benchmark: str
    image_path: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.example_id = _require_non_empty_text(self.example_id, "example_id")
        self.benchmark = _require_non_empty_text(self.benchmark, "benchmark")
        if self.benchmark not in ALLOWED_BENCHMARKS:
            raise SchemaValidationError(
                f"Unsupported benchmark `{self.benchmark}`. "
                f"Expected one of {sorted(ALLOWED_BENCHMARKS)}."
            )
        self.image_path = _require_non_empty_text(self.image_path, "image_path")
        self.question = _require_non_empty_text(self.question, "question")
        self.answer = _require_non_empty_text(self.answer, "answer")
        self.metadata = _require_mapping(self.metadata, "metadata")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizedExample":
        return cls(
            example_id=payload.get("example_id"),
            benchmark=payload.get("benchmark"),
            image_path=payload.get("image_path"),
            question=payload.get("question"),
            answer=payload.get("answer"),
            metadata=_require_mapping(payload.get("metadata"), "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "benchmark": self.benchmark,
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TraceStep:
    """One structured step in a reasoning trace."""

    image: str
    question: str
    step_id: int
    step_text: str
    step_type: str
    grounding_ref: str
    evidence_value: str
    label: int
    error_type: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = _require_non_empty_text(self.image, "image")
        self.question = _require_non_empty_text(self.question, "question")
        if not isinstance(self.step_id, int) or self.step_id <= 0:
            raise SchemaValidationError("`step_id` must be a positive integer.")
        self.step_text = _require_non_empty_text(self.step_text, "step_text")
        self.step_type = _require_non_empty_text(self.step_type, "step_type")
        if self.step_type not in ALLOWED_STEP_TYPES:
            raise SchemaValidationError(
                f"Unsupported step type `{self.step_type}`. "
                f"Expected one of {sorted(ALLOWED_STEP_TYPES)}."
            )
        self.grounding_ref = str(self.grounding_ref or "").strip()
        self.evidence_value = str(self.evidence_value or "").strip()
        self.label = _coerce_label(self.label)
        self.error_type = _require_non_empty_text(self.error_type, "error_type")
        if self.error_type not in ALLOWED_ERROR_TYPES:
            raise SchemaValidationError(
                f"Unsupported error type `{self.error_type}`. "
                f"Expected one of {sorted(ALLOWED_ERROR_TYPES)}."
            )
        self.metadata = _require_mapping(self.metadata, "metadata")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        default_image: str = "",
        default_question: str = "",
    ) -> "TraceStep":
        image = str(payload.get("image") or default_image).strip()
        question = str(payload.get("question") or default_question).strip()
        return cls(
            image=image,
            question=question,
            step_id=payload.get("step_id"),
            step_text=payload.get("step_text"),
            step_type=payload.get("step_type"),
            grounding_ref=payload.get("grounding_ref", ""),
            evidence_value=payload.get("evidence_value", ""),
            label=payload.get("label", 1),
            error_type=payload.get("error_type", "none"),
            metadata=_require_mapping(payload.get("metadata"), "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "question": self.question,
            "step_id": self.step_id,
            "step_text": self.step_text,
            "step_type": self.step_type,
            "grounding_ref": self.grounding_ref,
            "evidence_value": self.evidence_value,
            "label": self.label,
            "error_type": self.error_type,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TraceRecord:
    """Full structured trace attached to one normalized example."""

    trace_id: str
    example_id: str
    benchmark: str
    image_path: str
    question: str
    gold_answer: str
    predicted_answer: str | None
    steps: list[TraceStep]
    trace_mode: str = "canonical"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trace_id = _require_non_empty_text(self.trace_id, "trace_id")
        self.example_id = _require_non_empty_text(self.example_id, "example_id")
        self.benchmark = _require_non_empty_text(self.benchmark, "benchmark")
        if self.benchmark not in ALLOWED_BENCHMARKS:
            raise SchemaValidationError(
                f"Unsupported benchmark `{self.benchmark}`. "
                f"Expected one of {sorted(ALLOWED_BENCHMARKS)}."
            )
        self.image_path = _require_non_empty_text(self.image_path, "image_path")
        self.question = _require_non_empty_text(self.question, "question")
        self.gold_answer = _require_non_empty_text(self.gold_answer, "gold_answer")
        if self.predicted_answer is not None:
            self.predicted_answer = str(self.predicted_answer).strip() or None
        if self.trace_mode not in ALLOWED_TRACE_MODES:
            raise SchemaValidationError(
                f"Unsupported trace mode `{self.trace_mode}`. "
                f"Expected one of {sorted(ALLOWED_TRACE_MODES)}."
            )
        if not isinstance(self.steps, list) or not self.steps:
            raise SchemaValidationError("`steps` must be a non-empty list.")
        for step in self.steps:
            if not isinstance(step, TraceStep):
                raise SchemaValidationError("`steps` must contain `TraceStep` values.")
            if step.image != self.image_path:
                step.image = self.image_path
            if step.question != self.question:
                step.question = self.question
        expected_ids = list(range(1, len(self.steps) + 1))
        actual_ids = [step.step_id for step in self.steps]
        if actual_ids != expected_ids:
            raise SchemaValidationError(
                f"`steps` must have contiguous step ids starting at 1, got {actual_ids}."
            )
        self.metadata = _require_mapping(self.metadata, "metadata")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceRecord":
        image_path = payload.get("image_path")
        question = payload.get("question")
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            raise SchemaValidationError("`steps` must be a list.")
        steps = [
            TraceStep.from_dict(
                step_payload,
                default_image=str(image_path or ""),
                default_question=str(question or ""),
            )
            for step_payload in raw_steps
        ]
        return cls(
            trace_id=payload.get("trace_id"),
            example_id=payload.get("example_id"),
            benchmark=payload.get("benchmark"),
            image_path=image_path,
            question=question,
            gold_answer=payload.get("gold_answer"),
            predicted_answer=payload.get("predicted_answer"),
            steps=steps,
            trace_mode=payload.get("trace_mode", "canonical"),
            metadata=_require_mapping(payload.get("metadata"), "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "example_id": self.example_id,
            "benchmark": self.benchmark,
            "image_path": self.image_path,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "trace_mode": self.trace_mode,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }

