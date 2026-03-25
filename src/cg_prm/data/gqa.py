"""GQA adapter that normalizes questions and scene graphs into shared manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cg_prm.data.schema import NormalizedExample, read_json, write_jsonl


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
                {**dict(item), "__key__": str(item_key)}
                for item_key, item in value.items()
                if isinstance(item, Mapping)
            ]
    return [
        {**dict(item), "__key__": str(item_key)}
        for item_key, item in payload.items()
        if isinstance(item, Mapping)
    ]


def _maybe_join_image_path(image_root: str | Path, image_name: str) -> str:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(image_root) / image_name)


def _normalize_image_name(image_identifier: str) -> str:
    image_name = image_identifier.strip()
    if not image_name:
        raise ValueError("GQA example is missing an image identifier.")
    if Path(image_name).suffix:
        return image_name
    return f"{image_name}.jpg"


def _coerce_object_id(value: Any, fallback_index: int) -> str:
    text = str(value if value is not None else fallback_index).strip()
    if not text:
        return str(fallback_index)
    return text


def _coerce_attributes(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _coerce_relations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    relations: list[dict[str, str]] = []
    for relation in value:
        if not isinstance(relation, Mapping):
            continue
        name = str(relation.get("name") or relation.get("relation") or "").strip()
        target = str(
            relation.get("object")
            or relation.get("target")
            or relation.get("object_id")
            or relation.get("target_object_id")
            or ""
        ).strip()
        if not name or not target:
            continue
        relations.append({"name": name, "target_object_id": target})
    return relations


def _normalize_scene(scene: Mapping[str, Any], scene_key: str | None = None) -> dict[str, Any]:
    raw_objects = scene.get("objects", {})
    if isinstance(raw_objects, Mapping):
        object_items = list(raw_objects.items())
    elif isinstance(raw_objects, list):
        object_items = [(str(index), item) for index, item in enumerate(raw_objects)]
    else:
        object_items = []

    objects: list[dict[str, Any]] = []
    for fallback_index, (object_key, raw_object) in enumerate(object_items):
        if not isinstance(raw_object, Mapping):
            continue
        object_id = _coerce_object_id(
            raw_object.get("object_id")
            or raw_object.get("objectId")
            or object_key,
            fallback_index,
        )
        bbox = raw_object.get("bbox")
        if not isinstance(bbox, list):
            x = raw_object.get("x")
            y = raw_object.get("y")
            w = raw_object.get("w") or raw_object.get("width")
            h = raw_object.get("h") or raw_object.get("height")
            bbox = [x, y, w, h] if all(value is not None for value in (x, y, w, h)) else []
        objects.append(
            {
                "object_id": object_id,
                "name": str(raw_object.get("name") or raw_object.get("label") or "").strip(),
                "attributes": _coerce_attributes(raw_object.get("attributes")),
                "bbox": [float(value) for value in bbox] if isinstance(bbox, list) else [],
                "relations": _coerce_relations(raw_object.get("relations")),
            }
        )

    image_id = str(
        scene.get("image_id")
        or scene.get("imageId")
        or scene.get("image_filename")
        or scene_key
        or ""
    ).strip()
    return {
        "image_id": image_id,
        "width": scene.get("width"),
        "height": scene.get("height"),
        "location": scene.get("location"),
        "weather": scene.get("weather"),
        "objects": objects,
    }


def build_scene_lookup(scene_payload: Any) -> dict[str, dict[str, Any]]:
    """Build a scene-graph lookup keyed by image id and image filename."""
    lookup: dict[str, dict[str, Any]] = {}
    if scene_payload is None:
        return lookup
    if isinstance(scene_payload, Mapping):
        scenes = [
            (str(scene_key), scene_value)
            for scene_key, scene_value in scene_payload.items()
            if isinstance(scene_value, Mapping)
        ]
    else:
        scenes = [
            (None, scene_value)  # type: ignore[arg-type]
            for scene_value in _extract_records(scene_payload, ("scenes", "data"))
        ]
    for scene_key, scene in scenes:
        normalized = _normalize_scene(scene, scene_key=scene_key)
        if normalized["image_id"]:
            lookup[normalized["image_id"]] = normalized
            lookup[_normalize_image_name(normalized["image_id"])] = normalized
    return lookup


def _resolve_question_id(question_item: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("question_id", "questionId", "id", "__key__"):
        value = question_item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback_index)


def _resolve_image_identifier(question_item: Mapping[str, Any]) -> str:
    for key in ("imageId", "image_id", "image", "image_filename", "imageFilename"):
        value = question_item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("GQA question entry is missing an image identifier.")


def normalize_gqa_example(
    question_item: Mapping[str, Any],
    *,
    image_root: str | Path,
    split: str,
    fallback_index: int,
    scene_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> NormalizedExample:
    """Normalize one GQA question into the shared manifest schema."""
    question_id = _resolve_question_id(question_item, fallback_index)
    image_identifier = _resolve_image_identifier(question_item)
    image_name = _normalize_image_name(image_identifier)
    answer = str(question_item.get("answer") or question_item.get("fullAnswer") or "").strip()
    if not answer:
        raise ValueError(f"GQA question `{question_id}` does not contain an answer.")

    scene = None
    if scene_lookup:
        scene = scene_lookup.get(image_identifier) or scene_lookup.get(image_name)

    metadata = {
        "split": split,
        "source_id": question_id,
        "question_id": question_id,
        "image_id": image_identifier,
        "image_name": image_name,
        "semantic": question_item.get("semantic"),
        "types": question_item.get("types"),
        "annotations": question_item.get("annotations"),
        "scene_graph": dict(scene) if scene else {},
    }
    return NormalizedExample(
        example_id=f"gqa_{split}_{question_id}",
        benchmark="gqa",
        image_path=_maybe_join_image_path(image_root, image_name),
        question=str(question_item.get("question") or "").strip(),
        answer=answer,
        metadata=metadata,
    )


def load_gqa_manifest(
    questions_path: str | Path,
    *,
    image_root: str | Path,
    scene_graphs_path: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Load a GQA split and normalize it into the shared example schema."""
    payload = read_json(questions_path)
    question_items = _extract_records(payload, ("questions", "data"))
    default_split = split or str(
        payload.get("split") if isinstance(payload, Mapping) else ""
    ).strip() or Path(questions_path).stem
    scene_lookup = build_scene_lookup(read_json(scene_graphs_path)) if scene_graphs_path else {}
    return [
        normalize_gqa_example(
            item,
            image_root=image_root,
            split=default_split,
            fallback_index=index,
            scene_lookup=scene_lookup,
        )
        for index, item in enumerate(question_items)
    ]


def write_gqa_manifest(
    output_path: str | Path,
    questions_path: str | Path,
    *,
    image_root: str | Path,
    scene_graphs_path: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Normalize a GQA split and write the resulting JSONL manifest."""
    manifest = load_gqa_manifest(
        questions_path,
        image_root=image_root,
        scene_graphs_path=scene_graphs_path,
        split=split,
    )
    write_jsonl(output_path, manifest)
    return manifest
