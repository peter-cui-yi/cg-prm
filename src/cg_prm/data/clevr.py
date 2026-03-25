"""CLEVR adapter that normalizes questions and scenes into shared manifests."""

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
    raise ValueError(f"Could not find any of keys {list(preferred_keys)} in payload.")


def _maybe_join_image_path(image_root: str | Path, image_name: str) -> str:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(image_root) / image_name)


def _normalize_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    objects = []
    for index, obj in enumerate(scene.get("objects", [])):
        if not isinstance(obj, Mapping):
            continue
        objects.append(
            {
                "object_id": index,
                "color": obj.get("color"),
                "shape": obj.get("shape"),
                "size": obj.get("size"),
                "material": obj.get("material"),
                "position": obj.get("3d_coords") or obj.get("position"),
                "rotation": obj.get("rotation"),
                "pixel_coords": obj.get("pixel_coords"),
            }
        )
    relationships = scene.get("relationships") if isinstance(scene.get("relationships"), Mapping) else {}
    return {
        "image_index": scene.get("image_index"),
        "image_filename": scene.get("image_filename"),
        "split": scene.get("split"),
        "objects": objects,
        "relationships": dict(relationships),
    }


def build_scene_lookup(scene_payload: Any) -> dict[str, dict[str, Any]]:
    """Build a scene lookup keyed by image filename and image index."""
    lookup: dict[str, dict[str, Any]] = {}
    if scene_payload is None:
        return lookup
    scenes = _extract_records(scene_payload, ("scenes", "data"))
    for scene in scenes:
        normalized = _normalize_scene(scene)
        image_filename = normalized.get("image_filename")
        image_index = normalized.get("image_index")
        if image_filename is not None:
            lookup[str(image_filename)] = normalized
        if image_index is not None:
            lookup[str(image_index)] = normalized
    return lookup


def normalize_clevr_example(
    question_item: Mapping[str, Any],
    *,
    image_root: str | Path,
    split: str,
    fallback_index: int,
    scene_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> NormalizedExample:
    """Normalize one CLEVR question into the shared manifest schema."""
    question_id = str(
        question_item.get("question_index")
        or question_item.get("question_id")
        or fallback_index
    ).strip()
    image_name = str(question_item.get("image_filename") or "").strip()
    if not image_name:
        raise ValueError(f"CLEVR question `{question_id}` does not contain `image_filename`.")

    scene = None
    if scene_lookup:
        scene = scene_lookup.get(image_name)
        if scene is None and question_item.get("image_index") is not None:
            scene = scene_lookup.get(str(question_item["image_index"]))

    metadata = {
        "split": split,
        "source_id": question_id,
        "question_family_index": question_item.get("question_family_index"),
        "program": question_item.get("program"),
        "image_filename": image_name,
        "image_index": question_item.get("image_index"),
        "scene": dict(scene) if scene else {},
    }
    return NormalizedExample(
        example_id=f"clevr_{split}_{question_id}",
        benchmark="clevr",
        image_path=_maybe_join_image_path(image_root, image_name),
        question=str(question_item.get("question") or "").strip(),
        answer=str(question_item.get("answer") or "").strip(),
        metadata=metadata,
    )


def load_clevr_manifest(
    questions_path: str | Path,
    *,
    image_root: str | Path,
    scenes_path: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Load a CLEVR split and normalize it into the shared example schema."""
    payload = read_json(questions_path)
    question_items = _extract_records(payload, ("questions", "data"))
    default_split = split or str(
        payload.get("split") if isinstance(payload, Mapping) else ""
    ).strip() or Path(questions_path).stem
    scene_lookup = build_scene_lookup(read_json(scenes_path)) if scenes_path else {}
    return [
        normalize_clevr_example(
            item,
            image_root=image_root,
            split=default_split,
            fallback_index=index,
            scene_lookup=scene_lookup,
        )
        for index, item in enumerate(question_items)
    ]


def write_clevr_manifest(
    output_path: str | Path,
    questions_path: str | Path,
    *,
    image_root: str | Path,
    scenes_path: str | Path | None = None,
    split: str | None = None,
) -> list[NormalizedExample]:
    """Normalize a CLEVR split and write the resulting JSONL manifest."""
    manifest = load_clevr_manifest(
        questions_path,
        image_root=image_root,
        scenes_path=scenes_path,
        split=split,
    )
    write_jsonl(output_path, manifest)
    return manifest

