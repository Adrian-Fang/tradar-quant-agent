"""Small embedding-based retrieval over research records."""

from __future__ import annotations

import math
from typing import Any

from .loader import load_research_records


def _record_text(record: dict[str, Any]) -> str:
    return "\n".join((record["title"], record["text"]))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embeddings must be non-empty vectors of equal length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def prepare_semantic_corpus(
    *,
    embedder: Any,
    records: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    records = load_research_records() if records is None else records
    embeddings = embedder([_record_text(record) for record in records])
    if len(embeddings) != len(records):
        raise ValueError("embedder returned an unexpected number of vectors")
    return {"records": records, "embeddings": embeddings}


def retrieve_semantic(
    query: str,
    *,
    market: str | None = None,
    topic: str | None = None,
    status: str | None = None,
    limit: int = 5,
    embedder: Any = None,
    prepared_corpus: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not query.strip():
        return []

    if prepared_corpus is None:
        records = load_research_records()
        embeddings = None
    else:
        records = prepared_corpus["records"]
        embeddings = prepared_corpus["embeddings"]
        if len(records) != len(embeddings):
            raise ValueError("prepared corpus records and embeddings must match")

    filtered_records = []
    filtered_embeddings = []
    for index, record in enumerate(records):
        metadata = record["metadata"]
        if market is not None and str(metadata.get("market", "")).casefold() != market.casefold():
            continue
        if topic is not None and str(metadata.get("topic", "")).casefold() != topic.casefold():
            continue
        if status is not None and str(metadata.get("status", "")).casefold() != status.casefold():
            continue
        filtered_records.append(record)
        if embeddings is not None:
            filtered_embeddings.append(embeddings[index])

    if not filtered_records:
        return []

    if embedder is None:
        raise ValueError("embedder is required for semantic retrieval")
    if embeddings is None:
        texts = [query] + [_record_text(record) for record in filtered_records]
        vectors = embedder(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedder returned an unexpected number of vectors")
        query_vector = vectors[0]
        document_vectors = vectors[1:]
    else:
        vectors = embedder([query])
        if len(vectors) != 1:
            raise ValueError("embedder returned an unexpected number of query vectors")
        query_vector = vectors[0]
        document_vectors = filtered_embeddings

    matches = []
    for record, vector in zip(filtered_records, document_vectors):
        result = dict(record)
        result["score"] = _cosine_similarity(query_vector, vector)
        matches.append(result)

    matches.sort(key=lambda record: (-record["score"], record["research_id"], record["path"]))
    return matches[:limit]


__all__ = ["prepare_semantic_corpus", "retrieve_semantic"]
