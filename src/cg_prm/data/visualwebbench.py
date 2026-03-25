"""VisualWebBench adapter with weak-verification metadata normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cg_prm.data.schema import NormalizedExample, normalize_text, read_json, read_jsonl, write_jsonl


def _extract_records(payload: Any, preferred_keys: Iterable[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a list or mapping payload.")
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            return [
                {**dict(item), "__group__": str(item_key)}
                for item_key, item in value.items()
                if isinstance(item, Mapping)
            ]
    flattened: list[dict[str, Any]] = []
    for group_name, value in payload.items():
        if isinstance(value, list):
            flattened.extend(
                [{**dict(item), "__group__": str(group_name)} for item in value if isinstance(item, Mapping)]
            )
    if flattened:
        return flattened
    return [
        {**dict(item), "__group__": str(item_key)}
        for item_key, item in payload.items()
        if isinstance(item, Mapping)
    ]


def _maybe_join_image_path(image_root: str | Path | None, image_name: str) -> str:
    image_path = Path(image_name)
    if image_path.is_absolute() or image_name.startswith("http://") or image_name.startswith("https://"):
        return str(image_path) if image_path.is_absolute() else image_name
    if image_root is None:
        return image_name
    return str(Path(image_root) / image_name)


def _coerce_bbox(value: Any) -> list[float]:
    if isinstance(value, list):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return []
    if isinstance(value, Mapping):
        x = value.get("x")
        y = value.get("y")
        w = value.get("w") or value.get("width")
        h = value.get("h") or value.get("height")
        if all(item is not None for item in (x, y, w, h)):
            try:
                return [float(x), float(y), float(w), float(h)]
            except (TypeError, ValueError):
                return []
    return []


def _normalize_element(raw_element: Mapping[str, Any], fallback_id: str) -> dict[str, Any]:
    element_id = str(
        raw_element.get("element_id")
        or raw_element.get("elem_id")
        or raw_element.get("id")
        or raw_element.get("target_id")
        or fallback_id
    ).strip()
    text = str(
        raw_element.get("text")
        or raw_element.get("label")
        or raw_element.get("description")
        or raw_element.get("elem_desc")
        or raw_element.get("content")
        or ""
    ).strip()
    return {
        "element_id": element_id,
        "text": text,
        "normalized_text": normalize_text(text),
        "bbox": _coerce_bbox(raw_element.get("bbox") or raw_element.get("boundingBox")),
        "role": str(raw_element.get("role") or raw_element.get("type") or raw_element.get("tag") or "").strip(),
    }


def _extract_elements(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for key in ("elements", "candidate_elements", "page_elements", "ui_elements", "targets"):
        raw_elements = item.get(key)
        if not isinstance(raw_elements, list):
            continue
        for index, raw_element in enumerate(raw_elements):
            if isinstance(raw_element, Mapping):
                elements.append(_normalize_element(raw_element, f"{key}_{index}"))
            elif raw_element is not None and str(raw_element).strip():
                elements.append(
                    {
                        "element_id": f"{key}_{index}",
                        "text": str(raw_element).strip(),
                        "normalized_text": normalize_text(raw_element),
                        "bbox": [],
                        "role": "",
                    }
                )

    if not elements:
        bbox = item.get("bbox")
        elem_desc = item.get("elem_desc") or item.get("element_description") or item.get("target_description")
        if bbox is not None or (elem_desc is not None and str(elem_desc).strip()):
            elements.append(
                {
                    "element_id": "target",
                    "text": str(elem_desc or "").strip(),
                    "normalized_text": normalize_text(elem_desc or ""),
                    "bbox": _coerce_bbox(bbox),
                    "role": "target",
                }
            )
    return elements


def _resolve_example_id(item: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("id", "example_id", "question_id", "sample_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback_index)


def _resolve_question(item: Mapping[str, Any]) -> str:
    for key in ("question", "instruction", "prompt", "query", "task"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    task_type = str(item.get("task_type") or item.get("taskType") or "web task").strip()
    website = str(item.get("website") or "").strip()
    return f"{task_type} on {website}".strip()


def _resolve_answer(item: Mapping[str, Any]) -> str:
    for key in ("answer", "target_action", "action", "label", "target"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("VisualWebBench item does not contain a deterministic answer or action label.")


def _resolve_image_name(item: Mapping[str, Any]) -> str:
    for key in ("image", "image_path", "image_filename", "screenshot", "screenshot_path", "file_name"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("VisualWebBench item is missing an image path.")


def load_visualwebbench_manifest(
    items_path: str | Path,
    *,
    image_root: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Load a VisualWebBench-style file and normalize it into the shared schema."""
    path = Path(items_path)
    payload: Any
    if path.suffix.lower() == ".jsonl":
        payload = read_jsonl(path)
    else:
        payload = read_json(path)
    items = _extract_records(payload, ("data", "examples", "items"))
    default_split = split or str(
        payload.get("split") if isinstance(payload, Mapping) else ""
    ).strip() or path.stem

    manifest: list[NormalizedExample] = []
    for index, item in enumerate(items):
        example_id = _resolve_example_id(item, index)
        image_name = _resolve_image_name(item)
        elements = _extract_elements(item)
        verification_mode = "element" if elements else "weak_answer"
        metadata = {
            "split": default_split,
            "source_id": example_id,
            "task_type": item.get("task_type") or item.get("taskType"),
            "website": item.get("website"),
            "options": item.get("options"),
            "instruction": item.get("instruction"),
            "elements": elements,
            "verification_mode": verification_mode,
            "group": item.get("__group__"),
        }
        manifest.append(
            NormalizedExample(
                example_id=f"visualwebbench_{default_split}_{example_id}",
                benchmark="visualwebbench",
                image_path=_maybe_join_image_path(image_root, image_name),
                question=_resolve_question(item),
                answer=_resolve_answer(item),
                metadata=metadata,
            )
        )
    return manifest


def write_visualwebbench_manifest(
    output_path: str | Path,
    items_path: str | Path,
    *,
    image_root: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Normalize a VisualWebBench file and write the resulting JSONL manifest."""
    manifest = load_visualwebbench_manifest(
        items_path,
        image_root=image_root,
        split=split,
    )
    write_jsonl(output_path, manifest)
    return manifest
