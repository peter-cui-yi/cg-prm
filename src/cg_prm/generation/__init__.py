"""Generation-time utilities for trace creation and preprocessing."""

from cg_prm.generation.prompts import (
    PromptTemplate,
    get_prompt_template,
    list_prompt_templates,
    render_prompt,
)
from cg_prm.generation.segmentation import segment_trace
from cg_prm.generation.teacher import (
    GenerationConfig,
    TeacherRequest,
    TeacherOutput,
    build_teacher_request,
    build_teacher_requests,
    parse_teacher_output,
)

__all__ = [
    "GenerationConfig",
    "PromptTemplate",
    "TeacherRequest",
    "TeacherOutput",
    "build_teacher_request",
    "build_teacher_requests",
    "get_prompt_template",
    "list_prompt_templates",
    "parse_teacher_output",
    "render_prompt",
    "segment_trace",
]
