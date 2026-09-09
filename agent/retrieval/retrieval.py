"""Simple deterministic lexical retrieval over research records."""

from __future__ import annotations

import re
from typing import Any

from .loader import load_research_records


TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
FIELD_WEIGHTS = {
    "filename": 5,
    "title": 4,
    "question": 4,
    "tags": 3,
    "body": 1,
}


def _tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.casefold()))


def retrieve(
    query: str,
    *,
    market: str | None = None,
    topic: str | None = None,
    status: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")

    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    matches = []
    for record in load_research_records():
        metadata = record["metadata"]
        if market is not None and str(metadata.get("market", "")).casefold() != market.casefold():
            continue
        if topic is not None and str(metadata.get("topic", "")).casefold() != topic.casefold():
            continue
        if status is not None and str(metadata.get("status", "")).casefold() != status.casefold():
            continue

        fields = {
            "filename": record["path"],
            "title": record["title"],
            "question": record["question"],
            "tags": " ".join(record["tags"]),
            "body": record["text"],
        }
        score = sum(
            weight
            for field, weight in FIELD_WEIGHTS.items()
            if query_tokens & _tokens(fields[field])
        )
        if score:
            result = dict(record)
            result["score"] = score
            matches.append(result)

    matches.sort(key=lambda record: (-record["score"], record["research_id"], record["path"]))
    return matches[:limit]


__all__ = ["retrieve"]
