"""Provider-agnostic runtime verification for retrieved research records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.resources import load_prompt
from .semantic_retriever import retrieve_semantic


PROMPT = load_prompt("prompts/relevance_verification.md")
RECORD_EVIDENCE_FIELDS = (
    "research_id", "path", "metadata", "title", "question", "tags", "text",
    "source", "source_ref", "provenance", "source_hash", "schema_version",
)


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def _parse_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON must be an object"
    if set(parsed) != {"supported", "reason"}:
        return None, "response must contain exactly supported and reason"
    if not isinstance(parsed["supported"], bool):
        return None, "response.supported must be a boolean"
    if not isinstance(parsed["reason"], str):
        return None, "response.reason must be a string"
    return parsed, None


def verify_record(
    query: str,
    record: dict[str, Any],
    *,
    client: Any,
    model: str = "",
) -> dict[str, Any]:
    payload = {
        "model": model,
        "instructions": PROMPT,
        "input": json.dumps(
            {"query": query, "research_record": {key: record[key] for key in RECORD_EVIDENCE_FIELDS if key in record}},
            ensure_ascii=False,
        ),
    }
    try:
        response = client.create(payload)
        parsed, parse_error = _parse_response(_response_text(response))
    except Exception as exc:
        return {
            "status": "error",
            "supported": None,
            "reason": "",
            "error_type": "provider_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if parse_error:
        return {
            "status": "error",
            "supported": None,
            "reason": "",
            "error_type": "malformed_response",
            "error": parse_error,
        }

    supported = parsed["supported"]
    return {
        "status": "supported" if supported else "unsupported",
        "supported": supported,
        "reason": parsed["reason"],
        "error_type": None,
        "error": "",
    }


def retrieve_verified(
    query: str,
    *,
    client: Any,
    model: str = "",
    candidate_limit: int = 5,
    market: str | None = None,
    topic: str | None = None,
    status: str | None = None,
    embedder: Any = None,
    prepared_corpus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = retrieve_semantic(
        query,
        market=market,
        topic=topic,
        status=status,
        limit=candidate_limit,
        embedder=embedder,
        prepared_corpus=prepared_corpus,
    )
    return verify_candidates(query, candidates, client=client, model=model)


def verify_candidates(query: str, candidates: list[dict[str, Any]], *, client: Any, model: str = "") -> dict[str, Any]:
    """Verify full records from any retriever, retaining the accepted input order."""
    if not candidates:
        return {
            "status": "abstain",
            "results": [],
            "rejected": [],
            "errors": [],
            "reason": "no retrieval candidates",
        }

    results = []
    rejected = []
    errors = []
    for candidate in candidates:
        verification = verify_record(query, candidate, client=client, model=model)
        if verification["status"] == "supported":
            result = dict(candidate)
            result["verification"] = {
                "supported": True,
                "reason": verification["reason"],
            }
            results.append(result)
        elif verification["status"] == "unsupported":
            rejected.append({
                "research_id": candidate["research_id"],
                "reason": verification["reason"],
            })
        else:
            errors.append({
                "research_id": candidate["research_id"],
                "error_type": verification["error_type"],
                "error": verification["error"],
            })

    if errors:
        return {
            "status": "error",
            "results": [],
            "rejected": rejected,
            "errors": errors,
            "reason": "verification failed closed",
        }
    if not results:
        return {
            "status": "abstain",
            "results": [],
            "rejected": rejected,
            "errors": [],
            "reason": "all candidates unsupported",
        }
    return {
        "status": "ok",
        "results": results,
        "rejected": rejected,
        "errors": [],
        "reason": "",
    }


__all__ = ["retrieve_verified", "verify_record", "verify_candidates"]
