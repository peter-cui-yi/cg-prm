"""DocVQA adapter that normalizes raw benchmark files into shared manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from cg_prm.data.schema import NormalizedExample, normalize_text, read_json, write_jsonl


@dataclass(slots=True)
class OCRSpan:
    """Standardized OCR span for DocVQA-style evidence alignment."""

    span_id: str
    text: str
    bbox: list[float]
    page: int | None = None
    tokens: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "text": self.text,
            "normalized_text": normalize_text(self.text),
            "bbox": self.bbox,
            "page": self.page,
            "tokens": self.tokens or [],
        }


def _extract_records(payload: Any, preferred_keys: Iterable[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a list or mapping payload.")
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    raise ValueError(f"Could not find any of keys {list(preferred_keys)} in payload.")


def _maybe_join_image_path(image_root: str | Path, image_name: str) -> str:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(image_root) / image_name)


def _coerce_answers(question_item: Mapping[str, Any]) -> list[str]:
    answers = question_item.get("answers")
    if isinstance(answers, list):
        cleaned = [str(answer).strip() for answer in answers if str(answer).strip()]
        if cleaned:
            return cleaned
    single = str(question_item.get("answer", "")).strip()
    return [single] if single else []


def _resolve_question_id(question_item: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("questionId", "question_id", "id"):
        value = question_item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback_index)


def _resolve_image_name(question_item: Mapping[str, Any]) -> str:
    for key in ("image", "image_path", "image_filename", "document"):
        value = question_item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("DocVQA question entry is missing an image filename.")


def _flatten_word_entries(container: Mapping[str, Any], prefix: str) -> list[OCRSpan]:
    spans: list[OCRSpan] = []
    words = container.get("words")
    if not isinstance(words, list):
        return spans
    for word_index, word in enumerate(words):
        if not isinstance(word, Mapping):
            continue
        text = str(word.get("text") or word.get("value") or "").strip()
        if not text:
            continue
        bbox = word.get("boundingBox") or word.get("bbox") or []
        spans.append(
            OCRSpan(
                span_id=f"{prefix}:{word_index}",
                text=text,
                bbox=[float(value) for value in bbox] if isinstance(bbox, list) else [],
                page=word.get("page"),
                tokens=text.split(),
            )
        )
    return spans


def _flatten_line_entries(container: Mapping[str, Any], prefix: str) -> list[OCRSpan]:
    spans: list[OCRSpan] = []
    lines = container.get("lines")
    if not isinstance(lines, list):
        return spans
    for line_index, line in enumerate(lines):
        if not isinstance(line, Mapping):
            continue
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        bbox = line.get("boundingBox") or line.get("bbox") or []
        spans.append(
            OCRSpan(
                span_id=f"{prefix}:{line_index}",
                text=text,
                bbox=[float(value) for value in bbox] if isinstance(bbox, list) else [],
                page=line.get("page"),
                tokens=text.split(),
            )
        )
    return spans


def extract_ocr_spans(ocr_item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract standardized OCR spans from several common DocVQA-style formats."""
    spans: list[OCRSpan] = []
    if isinstance(ocr_item.get("ocr_spans"), list):
        for index, span in enumerate(ocr_item["ocr_spans"]):
            if not isinstance(span, Mapping):
                continue
            text = str(span.get("text") or "").strip()
            if not text:
                continue
            bbox = span.get("bbox") or span.get("boundingBox") or []
            spans.append(
                OCRSpan(
                    span_id=str(span.get("span_id") or f"ocr_span:{index}"),
                    text=text,
                    bbox=[float(value) for value in bbox] if isinstance(bbox, list) else [],
                    page=span.get("page"),
                    tokens=span.get("tokens"),
                )
            )

    spans.extend(_flatten_word_entries(ocr_item, "ocr_word"))
    spans.extend(_flatten_line_entries(ocr_item, "ocr_line"))

    recognition_results = ocr_item.get("recognitionResults")
    if isinstance(recognition_results, list):
        for page_index, page in enumerate(recognition_results):
            if not isinstance(page, Mapping):
                continue
            page_prefix = f"ocr_page{page_index}"
            spans.extend(_flatten_line_entries(page, page_prefix))
            spans.extend(_flatten_word_entries(page, page_prefix))

    unique_spans: dict[str, OCRSpan] = {}
    for span in spans:
        if span.span_id not in unique_spans:
            unique_spans[span.span_id] = span
    return [span.to_dict() for span in unique_spans.values()]


def build_ocr_lookup(ocr_payload: Any) -> dict[str, dict[str, Any]]:
    """Build a flexible lookup keyed by question id and image name."""
    lookup: dict[str, dict[str, Any]] = {}
    if ocr_payload is None:
        return lookup
    entries = _extract_records(ocr_payload, ("data", "ocr_results", "documents", "annotations"))
    for entry in entries:
        candidate_keys = []
        for key in ("questionId", "question_id", "id", "image", "image_filename", "document"):
            value = entry.get(key)
            if value is not None and str(value).strip():
                candidate_keys.append(str(value).strip())
        for candidate_key in candidate_keys:
            lookup[candidate_key] = entry
    return lookup


def normalize_docvqa_example(
    question_item: Mapping[str, Any],
    *,
    image_root: str | Path,
    split: str,
    fallback_index: int,
    ocr_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> NormalizedExample:
    """Normalize one raw DocVQA question into the shared manifest schema."""
    question_id = _resolve_question_id(question_item, fallback_index)
    image_name = _resolve_image_name(question_item)
    answers = _coerce_answers(question_item)
    if not answers:
        raise ValueError(f"DocVQA question `{question_id}` does not contain an answer.")

    ocr_item = None
    if ocr_lookup:
        ocr_item = ocr_lookup.get(question_id) or ocr_lookup.get(image_name)
    ocr_spans = extract_ocr_spans(ocr_item or {})

    metadata = {
        "split": split,
        "source_id": question_id,
        "question_id": question_id,
        "image_name": image_name,
        "answers": answers,
        "ocr_spans": ocr_spans,
    }
    if "ucsf_document_id" in question_item:
        metadata["document_id"] = question_item["ucsf_document_id"]
    return NormalizedExample(
        example_id=f"docvqa_{split}_{question_id}",
        benchmark="docvqa",
        image_path=_maybe_join_image_path(image_root, image_name),
        question=str(question_item.get("question") or "").strip(),
        answer=answers[0],
        metadata=metadata,
    )


def load_docvqa_manifest(
    questions_path: str | Path,
    *,
    image_root: str | Path,
    ocr_path: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Load a DocVQA split and normalize it into the shared example schema."""
    payload = read_json(questions_path)
    question_items = _extract_records(payload, ("data", "dataset", "questions"))
    default_split = split or str(
        payload.get("split") if isinstance(payload, Mapping) else ""
    ).strip() or Path(questions_path).stem
    ocr_lookup = build_ocr_lookup(read_json(ocr_path)) if ocr_path else {}
    return [
        normalize_docvqa_example(
            item,
            image_root=image_root,
            split=default_split,
            fallback_index=index,
            ocr_lookup=ocr_lookup,
        )
        for index, item in enumerate(question_items)
    ]


def write_docvqa_manifest(
    output_path: str | Path,
    questions_path: str | Path,
    *,
    image_root: str | Path,
    ocr_path: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Normalize a DocVQA split and write the resulting JSONL manifest."""
    manifest = load_docvqa_manifest(
        questions_path,
        image_root=image_root,
        ocr_path=ocr_path,
        split=split,
    )
    write_jsonl(output_path, manifest)
    return manifest

