"""Provider-agnostic runtime verification for retrieved research records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.resources import load_prompt
from ..core.providers import provider_error_type
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
    error = _decision_error(parsed)
    return (None, error) if error else (parsed, None)


def _decision_error(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return "response JSON must be an object"
    if set(parsed) != {"supported", "reason"}:
        return "response must contain exactly supported and reason"
    if not isinstance(parsed["supported"], bool):
        return "response.supported must be a boolean"
    if not isinstance(parsed["reason"], str):
        return "response.reason must be a string"
    return None


def _verify_batch(query: str, candidates: list[dict[str, Any]], *, client: Any, model: str) -> list[dict[str, Any]]:
    ids = [record.get("research_id") for record in candidates]
    error_type, error = "invalid_candidates", "candidate research_ids must be unique nonempty strings"
    if all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids):
        payload = {
            "model": model, "instructions": PROMPT,
            "input": json.dumps({"query": query, "research_records": [
                {key: record[key] for key in RECORD_EVIDENCE_FIELDS if key in record}
                for record in candidates
            ]}, ensure_ascii=False),
        }
        try:
            response = client.create(payload)
        except Exception as exc:
            error_type, error = provider_error_type(exc), f"{type(exc).__name__}: {exc}"
        else:
            error_type = "malformed_response"
            try:
                parsed = json.loads(_response_text(response).strip())
                if not isinstance(parsed, dict) or set(parsed) != {"results"} or not isinstance(parsed["results"], list):
                    raise ValueError("response must contain exactly results (an array)")
                decisions = {}
                for index, row in enumerate(parsed["results"]):
                    if not isinstance(row, dict) or set(row) != {"research_id", "supported", "reason"}:
                        raise ValueError(f"response.results[{index}] must contain exactly research_id, supported and reason")
                    identity = row["research_id"]
                    if not isinstance(identity, str) or identity not in ids or identity in decisions:
                        raise ValueError(f"response.results[{index}].research_id is unknown or duplicated")
                    shape_error = _decision_error({key: row[key] for key in ("supported", "reason")})
                    if shape_error:
                        raise ValueError(f"response.results[{index}]: {shape_error}")
                    decisions[identity] = {**row, "status": "supported" if row["supported"] else "unsupported"}
                if set(decisions) != set(ids):
                    raise ValueError("response.results must include every input research_id exactly once")
                return [decisions[identity] for identity in ids]
            except (ValueError, TypeError) as exc:
                error = str(exc)
    # A partial batch can never publish a supported record, even if some rows parsed.
    return [{"status": "error", "error_type": error_type, "error": error} for _ in candidates]


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
            "error_type": provider_error_type(exc),
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
    verifications = (
        [verify_record(query, candidates[0], client=client, model=model)] if len(candidates) == 1
        else _verify_batch(query, candidates, client=client, model=model)
    )
    for candidate, verification in zip(candidates, verifications):
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
                "research_id": candidate.get("research_id"),
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
