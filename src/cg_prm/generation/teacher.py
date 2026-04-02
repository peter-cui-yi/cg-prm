"""Provider-agnostic teacher generation records and parsing helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from cg_prm.data.schema import NormalizedExample, TraceRecord, TraceStep
from cg_prm.generation.prompts import get_prompt_template, render_prompt
from cg_prm.generation.segmentation import segment_trace

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", flags=re.DOTALL)
_ANSWER_PATTERNS = (
    re.compile(r"(?:final answer|answer)\s*[:\-]\s*(.+)$", flags=re.IGNORECASE),
    re.compile(r"(?:therefore|thus|so)\s+the\s+answer\s+is\s+(.+)$", flags=re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation configuration attached to each teacher request."""

    model_name: str
    prompt_id: str
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationConfig":
        return cls(
            model_name=str(payload.get("model_name") or "").strip(),
            prompt_id=str(payload.get("prompt_id") or "").strip(),
            max_tokens=int(payload.get("max_tokens", 1024)),
            temperature=float(payload.get("temperature", 0.2)),
            top_p=float(payload.get("top_p", 0.95)),
            seed=int(payload.get("seed", 0)),
        )


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    """One rendered generation request for a normalized example."""

    example: NormalizedExample
    config: GenerationConfig
    prompt: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example.to_dict(),
            "config": self.config.to_dict(),
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherRequest":
        return cls(
            example=NormalizedExample.from_dict(payload["example"]),
            config=GenerationConfig.from_dict(payload["config"]),
            prompt=dict(payload["prompt"]),
        )


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    """Serializable raw generation output before verification."""

    request: TeacherRequest
    raw_text: str
    provider: str = "unknown"
    generation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "example_id": self.request.example.example_id,
            "benchmark": self.request.example.benchmark,
            "image_path": self.request.example.image_path,
            "question": self.request.example.question,
            "answer": self.request.example.answer,
            "example_metadata": self.request.example.metadata,
            "config": self.request.config.to_dict(),
            "provider": self.provider,
            "prompt": self.request.prompt,
            "raw_text": self.raw_text,
            "generation_metadata": self.generation_metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherOutput":
        example_payload = {
            "example_id": payload.get("example_id"),
            "benchmark": payload.get("benchmark"),
            "image_path": payload.get("image_path"),
            "question": payload.get("question"),
            "answer": payload.get("gold_answer")
            or payload.get("answer")
            or payload.get("reference_answer")
            or "",
            "metadata": payload.get("example_metadata") or {},
        }
        request_payload = payload.get("request")
        if isinstance(request_payload, dict):
            request = TeacherRequest.from_dict(request_payload)
        else:
            request = TeacherRequest(
                example=NormalizedExample.from_dict(example_payload),
                config=GenerationConfig.from_dict(payload["config"]),
                prompt=dict(payload.get("prompt") or {}),
            )
        return cls(
            request=request,
            raw_text=str(payload.get("raw_text") or ""),
            provider=str(payload.get("provider") or "unknown"),
            generation_metadata=dict(payload.get("generation_metadata") or {}),
        )


def build_teacher_request(
    example: NormalizedExample,
    config: GenerationConfig,
) -> TeacherRequest:
    """Render one teacher request using the configured prompt template."""
    template = get_prompt_template(config.prompt_id)
    prompt = render_prompt(template, example)
    return TeacherRequest(example=example, config=config, prompt=prompt)


def build_teacher_requests(
    examples: list[NormalizedExample],
    config: GenerationConfig,
) -> list[TeacherRequest]:
    """Render teacher requests for a batch of normalized examples."""
    return [build_teacher_request(example, config) for example in examples]


def _strip_code_fences(text: str) -> str:
    match = _JSON_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _maybe_parse_json_payload(text: str) -> dict[str, Any] | None:
    candidate = _strip_code_fences(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _infer_step_type(step_text: str, benchmark: str) -> str:
    lowered = step_text.lower()
    if any(token in lowered for token in ("locate", "find", "look at", "header", "field")):
        return "locate"
    if any(token in lowered for token in ("read", "extract", "text says", "shows")):
        return "read" if benchmark in {"docvqa", "visualwebbench"} else "extract"
    if any(token in lowered for token in ("count", "number of")):
        return "count"
    if any(token in lowered for token in ("left of", "right of", "behind", "front of", "relation")):
        return "relate"
    if any(token in lowered for token in ("click", "tap", "select", "press", "open")):
        return "derive"
    if any(token in lowered for token in ("therefore", "thus", "so the answer", "final answer")):
        return "answer"
    if any(token in lowered for token in ("because", "implies", "means")):
        return "reason"
    return "derive"


def _extract_predicted_answer(raw_text: str, segments: list[str]) -> str | None:
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            return match.group(1).strip(" .")
    if segments:
        last_segment = segments[-1]
        for pattern in _ANSWER_PATTERNS:
            match = pattern.search(last_segment)
            if match:
                return match.group(1).strip(" .")
    return None


def _build_trace_from_json(
    payload: dict[str, Any],
    output: TeacherOutput,
    trace_mode: str,
) -> TraceRecord:
    request = output.request
    raw_steps = payload.get("steps", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Structured teacher output must contain a non-empty `steps` list.")
    steps = [
        TraceStep.from_dict(
            step_payload,
            default_image=request.example.image_path,
            default_question=request.example.question,
        )
        for step_payload in raw_steps
    ]
    return TraceRecord(
        trace_id=(
            f"{request.example.example_id}__{request.config.prompt_id}"
            f"__seed{request.config.seed}"
        ),
        example_id=request.example.example_id,
        benchmark=request.example.benchmark,
        image_path=request.example.image_path,
        question=request.example.question,
        gold_answer=request.example.answer,
        predicted_answer=str(payload.get("predicted_answer") or "").strip() or None,
        steps=steps,
        trace_mode=trace_mode,
        metadata={
            "teacher": output.to_dict(),
            "parsing_mode": "json",
        },
    )


def _build_trace_from_text(
    output: TeacherOutput,
    *,
    trace_mode: str,
) -> TraceRecord:
    request = output.request
    raw_text = output.raw_text.strip()
    
    # Check if we have any actual text
    if not raw_text:
        raise ValueError(f"Teacher output for {request.example.example_id} has empty raw_text. Make sure vLLM inference saved the actual model response.")
    
    segments = segment_trace(raw_text, trace_mode)
    
    # If no segments found, split by newlines/paragraphs as fallback
    if not segments:
        # Split by double newlines (paragraphs) or single newlines
        segments = [s.strip() for s in raw_text.split('\n\n') if s.strip()]
        if not segments:
            segments = [s.strip() for s in raw_text.split('\n') if s.strip() and len(s.strip()) > 20]
        # If still nothing, use the whole text as one step
        if not segments and len(raw_text) > 50:
            segments = [raw_text]
    
    if not segments:
        raise ValueError(
            f"Teacher output for {request.example.example_id} could not be segmented into any steps. "
            f"raw_text length: {len(raw_text)}. "
            f"First 200 chars: {raw_text[:200]}"
        )
    steps = [
        TraceStep(
            image=request.example.image_path,
            question=request.example.question,
            step_id=index,
            step_text=segment,
            step_type=_infer_step_type(segment, request.example.benchmark),
            grounding_ref="",
            evidence_value="",
            label=1,
        )
        for index, segment in enumerate(segments, start=1)
    ]
    predicted_answer = _extract_predicted_answer(output.raw_text, segments)
    return TraceRecord(
        trace_id=(
            f"{request.example.example_id}__{request.config.prompt_id}"
            f"__seed{request.config.seed}"
        ),
        example_id=request.example.example_id,
        benchmark=request.example.benchmark,
        image_path=request.example.image_path,
        question=request.example.question,
        gold_answer=request.example.answer,
        predicted_answer=predicted_answer,
        steps=steps,
        trace_mode=trace_mode,
        metadata={
            "teacher": output.to_dict(),
            "parsing_mode": "text",
        },
    )


def parse_teacher_output(output: TeacherOutput) -> TraceRecord:
    """Parse raw teacher text into a shared `TraceRecord`.

    Structured JSON outputs are preferred for canonical prompts. Free-form or
    lightly structured outputs fall back to deterministic segmentation.
    """
    prompt_template = get_prompt_template(output.request.config.prompt_id)
    payload = _maybe_parse_json_payload(output.raw_text) if prompt_template.expects_json else None
    if payload is not None:
        return _build_trace_from_json(payload, output, prompt_template.trace_mode)
    return _build_trace_from_text(output, trace_mode=prompt_template.trace_mode)
