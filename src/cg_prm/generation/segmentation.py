"""Deterministic free-form trace segmentation for CG-PRM."""

from __future__ import annotations

import re

TRACE_MODES = frozenset({"canonical", "light", "free"})
DISCOURSE_MARKERS = (
    "first",
    "second",
    "third",
    "next",
    "then",
    "after that",
    "because",
    "therefore",
    "thus",
    "so",
    "however",
    "finally",
    "overall",
)

_NUMBERED_PREFIX = re.compile(
    r"^\s*(?:step\s*\d+\s*[:.)-]?|\d+\s*[:.)-]|[-*•])\s*",
    flags=re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9])")
_DISCOURSE_SPLIT = re.compile(
    rf",\s+(?=(?:{'|'.join(re.escape(marker) for marker in DISCOURSE_MARKERS)})\b)",
    flags=re.IGNORECASE,
)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_chunk(chunk: str) -> str:
    chunk = _NUMBERED_PREFIX.sub("", chunk.strip())
    chunk = re.sub(r"\s+", " ", chunk)
    return chunk.strip(" -;:,")


def _split_by_lines(text: str) -> list[str]:
    return [_clean_chunk(line) for line in text.splitlines() if _clean_chunk(line)]


def _split_sentences(text: str) -> list[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    chunks = _SENTENCE_BOUNDARY.split(text)
    return [_clean_chunk(chunk) for chunk in chunks if _clean_chunk(chunk)]


def _split_free_clauses(text: str) -> list[str]:
    sentence_level = _split_sentences(text)
    pieces: list[str] = []
    for sentence in sentence_level:
        clause_chunks = _DISCOURSE_SPLIT.split(sentence)
        for clause in clause_chunks:
            cleaned = _clean_chunk(clause)
            if cleaned:
                pieces.append(cleaned)
    return pieces


def segment_trace(text: str, mode: str) -> list[str]:
    """Segment a reasoning trace into deterministic step units.

    `canonical` prefers newline or numbered-list boundaries.
    `light` uses line boundaries, then sentence boundaries.
    `free` uses sentence boundaries plus a fixed discourse-marker split.
    """
    if mode not in TRACE_MODES:
        raise ValueError(f"Unsupported mode `{mode}`. Expected one of {sorted(TRACE_MODES)}.")

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    if mode == "canonical":
        line_chunks = _split_by_lines(normalized)
        if len(line_chunks) >= 2:
            return line_chunks
        return _split_sentences(normalized)

    if mode == "light":
        line_chunks = _split_by_lines(normalized)
        if len(line_chunks) >= 2:
            segmented: list[str] = []
            for chunk in line_chunks:
                segmented.extend(_split_sentences(chunk))
            return [piece for piece in segmented if piece]
        return _split_sentences(normalized)

    return _split_free_clauses(normalized)

