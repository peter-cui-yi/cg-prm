"""Teacher prompt registry for CG-PRM trace generation."""

from __future__ import annotations

from dataclasses import dataclass

from cg_prm.data.schema import ALLOWED_BENCHMARKS, ALLOWED_TRACE_MODES, NormalizedExample


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Prompt definition used for teacher trace generation."""

    prompt_id: str
    benchmark: str
    trace_mode: str
    expects_json: bool
    description: str
    system_prompt: str
    user_template: str

    def __post_init__(self) -> None:
        if self.benchmark not in ALLOWED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark `{self.benchmark}`. "
                f"Expected one of {sorted(ALLOWED_BENCHMARKS)}."
            )
        if self.trace_mode not in ALLOWED_TRACE_MODES:
            raise ValueError(
                f"Unsupported trace mode `{self.trace_mode}`. "
                f"Expected one of {sorted(ALLOWED_TRACE_MODES)}."
            )


def _canonical_json_schema(benchmark: str) -> str:
    grounding_hint = "ocr_span:<id>" if benchmark == "docvqa" else "object:<id>|objects:<id,id>|relation:<name>:<src>:<dst>"
    return (
        "Return valid JSON with this schema only:\n"
        "{{\n"
        '  "predicted_answer": "<answer>",\n'
        '  "steps": [\n'
        "    {{\n"
        '      "step_id": 1,\n'
        '      "step_text": "<reasoning step>",\n'
        '      "step_type": "<locate|read|extract|identify|relate|count|compute|derive|answer|reason|verify>",\n'
        f'      "grounding_ref": "<{grounding_hint}>",\n'
        '      "evidence_value": "<short evidence string>"\n'
        "    }}\n"
        "  ]\n"
        "}}\n"
        "Do not wrap the JSON in markdown fences."
    )


PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    "docvqa_canonical_v1": PromptTemplate(
        prompt_id="docvqa_canonical_v1",
        benchmark="docvqa",
        trace_mode="canonical",
        expects_json=True,
        description="Structured DocVQA trace with explicit OCR grounding references.",
        system_prompt=(
            "You are generating evidence-faithful reasoning traces for document question answering. "
            "Use only evidence visible in the image."
        ),
        user_template=(
            "Question: {question}\n"
            "Image path: {image_path}\n"
            "Task: Answer the question and provide a structured reasoning trace. "
            "Each grounding-critical step must cite an OCR span id when possible.\n\n"
            f"{_canonical_json_schema('docvqa')}"
        ),
    ),
    "docvqa_light_v1": PromptTemplate(
        prompt_id="docvqa_light_v1",
        benchmark="docvqa",
        trace_mode="light",
        expects_json=False,
        description="Lightly structured DocVQA reasoning in short numbered steps.",
        system_prompt=(
            "You are generating concise step-by-step document reasoning. "
            "Prefer grounded evidence over fluent filler."
        ),
        user_template=(
            "Question: {question}\n"
            "Image path: {image_path}\n"
            "Write 2-4 short numbered steps, then a final answer line. "
            "Mention concrete evidence text when possible."
        ),
    ),
    "docvqa_free_v1": PromptTemplate(
        prompt_id="docvqa_free_v1",
        benchmark="docvqa",
        trace_mode="free",
        expects_json=False,
        description="Natural free-form DocVQA reasoning without explicit schema fields.",
        system_prompt=(
            "Answer the document question naturally, but ensure the reasoning is visually grounded."
        ),
        user_template=(
            "Question: {question}\n"
            "Image path: {image_path}\n"
            "Give a natural explanation followed by the answer."
        ),
    ),
    "clevr_canonical_v1": PromptTemplate(
        prompt_id="clevr_canonical_v1",
        benchmark="clevr",
        trace_mode="canonical",
        expects_json=True,
        description="Structured CLEVR trace with explicit object or relation references.",
        system_prompt=(
            "You are generating evidence-faithful reasoning traces for compositional visual reasoning. "
            "Use only objects and relations that are visually supported."
        ),
        user_template=(
            "Question: {question}\n"
            "Image path: {image_path}\n"
            "Task: Answer the question and provide a structured reasoning trace. "
            "Use object ids or relation references when possible.\n\n"
            f"{_canonical_json_schema('clevr')}"
        ),
    ),
    "clevr_light_v1": PromptTemplate(
        prompt_id="clevr_light_v1",
        benchmark="clevr",
        trace_mode="light",
        expects_json=False,
        description="Lightly structured CLEVR reasoning in short numbered steps.",
        system_prompt=(
            "You are generating concise step-by-step compositional visual reasoning."
        ),
        user_template=(
            "Question: {question}\n"
            "Image path: {image_path}\n"
            "Write 2-4 short numbered steps, then a final answer line. "
            "Be explicit about counted objects or relations when relevant."
        ),
    ),
    "clevr_free_v1": PromptTemplate(
        prompt_id="clevr_free_v1",
        benchmark="clevr",
        trace_mode="free",
        expects_json=False,
        description="Natural free-form CLEVR reasoning without explicit schema fields.",
        system_prompt=(
            "Answer the visual reasoning question naturally while remaining visually faithful."
        ),
        user_template=(
            "Question: {question}\n"
            "Image path: {image_path}\n"
            "Give a natural explanation followed by the answer."
        ),
    ),
}


def list_prompt_templates(benchmark: str | None = None) -> list[PromptTemplate]:
    """List available prompt templates, optionally filtered by benchmark."""
    templates = list(PROMPT_REGISTRY.values())
    if benchmark is None:
        return sorted(templates, key=lambda template: template.prompt_id)
    if benchmark not in ALLOWED_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark `{benchmark}`. Expected one of {sorted(ALLOWED_BENCHMARKS)}."
        )
    return sorted(
        [template for template in templates if template.benchmark == benchmark],
        key=lambda template: template.prompt_id,
    )


def get_prompt_template(prompt_id: str) -> PromptTemplate:
    """Fetch one prompt template from the registry."""
    try:
        return PROMPT_REGISTRY[prompt_id]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt template `{prompt_id}`.") from exc


def render_prompt(template: PromptTemplate, example: NormalizedExample) -> dict[str, str]:
    """Render one teacher prompt for a normalized example."""
    if example.benchmark != template.benchmark:
        raise ValueError(
            f"Prompt `{template.prompt_id}` targets `{template.benchmark}`, "
            f"but example benchmark is `{example.benchmark}`."
        )
    return {
        "system": template.system_prompt,
        "user": template.user_template.format(
            question=example.question,
            image_path=example.image_path,
            example_id=example.example_id,
            benchmark=example.benchmark,
        ),
    }
