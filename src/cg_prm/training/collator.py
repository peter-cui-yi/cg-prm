"""Formatting and collation utilities for verifier-style SFT training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _trace_dict(trace: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    return dict(trace)


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("`torch` is required for tokenized collation.") from exc
    return torch


def serialize_trace(trace: Mapping[str, Any] | dict[str, Any]) -> str:
    """Serialize a trace into a stable human-readable multiline format."""
    payload = _trace_dict(trace)
    steps = payload.get("steps", [])
    lines = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_id = step.get("step_id", "?")
        step_type = step.get("step_type", "free")
        step_text = str(step.get("step_text") or "").strip()
        grounding_ref = str(step.get("grounding_ref") or "").strip() or "none"
        evidence_value = str(step.get("evidence_value") or "").strip() or "none"
        lines.append(
            f"{step_id}. [{step_type}] {step_text} | grounding_ref={grounding_ref} | evidence={evidence_value}"
        )
    return "\n".join(lines)


def format_pointwise_example(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one pointwise supervision record into prompt-target text."""
    trace = _trace_dict(record["trace"])
    question = trace["question"]
    image_path = trace["image_path"]
    trace_text = serialize_trace(trace)
    prompt = (
        "You are a multimodal verifier.\n"
        "Given an image, a question, and a reasoning trace, judge whether each step is grounded.\n"
        "Return JSON with `step_labels`, `final_score`, and `trace_label`.\n\n"
        f"Image path: {image_path}\n"
        f"Question: {question}\n"
        "Trace:\n"
        f"{trace_text}\n"
    )
    target = json.dumps(
        {
            "step_labels": list(record["step_labels"]),
            "final_score": float(record["trace_score_target"]),
            "trace_label": int(record["trace_label"]),
        },
        ensure_ascii=True,
    )
    return {
        "prompt": prompt,
        "target": target,
        "image_path": image_path,
        "record_id": record["record_id"],
    }


def format_pairwise_example(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one pairwise supervision record into prompt-target text."""
    preferred = _trace_dict(record["preferred_trace"])
    rejected = _trace_dict(record["rejected_trace"])
    question = preferred["question"]
    image_path = preferred["image_path"]
    preferred_text = serialize_trace(preferred)
    rejected_text = serialize_trace(rejected)
    prompt = (
        "You are a multimodal verifier.\n"
        "Given two reasoning traces for the same image and question, choose the more grounded trace.\n"
        "Respond with a JSON object containing `preferred_trace` set to `A` or `B`.\n\n"
        f"Image path: {image_path}\n"
        f"Question: {question}\n\n"
        "Trace A:\n"
        f"{preferred_text}\n\n"
        "Trace B:\n"
        f"{rejected_text}\n"
    )
    target = json.dumps({"preferred_trace": "A"}, ensure_ascii=True)
    return {
        "prompt": prompt,
        "target": target,
        "image_path": image_path,
        "record_id": record["pair_id"],
    }


@dataclass(slots=True)
class _BaseTraceCollator:
    """Shared text SFT collation logic for verifier training."""

    tokenizer: Any | None
    max_length: int = 4096
    prompt_template_suffix: str = "\nVerifier Output:\n"

    def _tokenize_supervised_texts(self, prompts: Sequence[str], targets: Sequence[str]) -> dict[str, Any]:
        if self.tokenizer is None:
            return {"prompts": list(prompts), "targets": list(targets)}
        torch = _load_torch()
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        joined_prompts = [prompt + self.prompt_template_suffix for prompt in prompts]
        full_texts = [prompt + target for prompt, target in zip(joined_prompts, targets)]

        model_inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_inputs = self.tokenizer(
            joined_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=False,
        )
        labels = model_inputs["input_ids"].clone()
        prompt_lengths = [len(ids) for ids in prompt_inputs["input_ids"]]
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100
        model_inputs["labels"] = labels
        return model_inputs


@dataclass(slots=True)
class PointwiseTraceCollator(_BaseTraceCollator):
    """Collator for pointwise verifier SFT examples."""

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        formatted = [format_pointwise_example(record) for record in batch]
        prompts = [item["prompt"] for item in formatted]
        targets = [item["target"] for item in formatted]
        payload = self._tokenize_supervised_texts(prompts, targets)
        payload["image_paths"] = [item["image_path"] for item in formatted]
        payload["record_ids"] = [item["record_id"] for item in formatted]
        return payload


@dataclass(slots=True)
class PairwiseTraceCollator(_BaseTraceCollator):
    """Collator for pairwise verifier SFT examples."""

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        formatted = [format_pairwise_example(record) for record in batch]
        prompts = [item["prompt"] for item in formatted]
        targets = [item["target"] for item in formatted]
        payload = self._tokenize_supervised_texts(prompts, targets)
        payload["image_paths"] = [item["image_path"] for item in formatted]
        payload["record_ids"] = [item["record_id"] for item in formatted]
        return payload
