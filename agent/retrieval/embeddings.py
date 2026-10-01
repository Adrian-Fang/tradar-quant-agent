"""Pinned Ollama model revision and bounded, non-truncating embedding requests."""

import json
import math
from urllib import request

from ..core.providers import OllamaEmbeddingClient


def validate_vectors(vectors: list, count: int, size: int = 1024) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("embedding batch count mismatch")
    result = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != size:
            raise ValueError(f"embedding must have {size} dimensions")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
            raise ValueError("embedding must contain finite numeric values")
        if not any(vector):
            raise ValueError("zero embedding is invalid for cosine similarity")
        result.append([float(value) for value in vector])
    return result


class OllamaDense:
    def __init__(self, profile: dict, *, base_url: str | None = None):
        self.client = OllamaEmbeddingClient(model=profile["dense"]["model"], base_url=base_url, truncate=False)

    @property
    def revision(self) -> str:
        with request.urlopen(f"{self.client.base_url}/api/tags", timeout=15) as response:
            payload = json.load(response)
        match = next((model for model in payload.get("models", []) if model.get("name") == self.client.model), None)
        if not match or not isinstance(match.get("digest"), str) or not match["digest"]:
            raise RuntimeError(f"embedding model absent; manually run: ollama pull {self.client.model}")
        return match["digest"]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 8):
            batch = texts[start:start + 8]
            vectors.extend(validate_vectors(self.client(batch), len(batch)))
        return vectors
