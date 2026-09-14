"""Provider-agnostic runtime verification of answer grounding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.resources import load_prompt


PROMPT = load_prompt("prompts/grounding.md")
LABELS = {"supported", "unsupported", "contradicted", "unverifiable"}


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def parse_grounding_response(
    text: str,
    evidence_ids: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON must be an object"
    if set(parsed) != {"answer", "claims"}:
        return None, "response must contain exactly answer and claims"
    if not isinstance(parsed["answer"], str):
        return None, "response.answer must be a string"
    if not isinstance(parsed["claims"], list):
        return None, "response.claims must be an array"
    for claim in parsed["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"claim", "evidence_ids", "grounding"}:
            return None, "each claim must contain exactly claim, evidence_ids, and grounding"
        if not isinstance(claim["claim"], str) or not claim["claim"].strip():
            return None, "claim.claim must be a non-empty string"
        if not isinstance(claim["evidence_ids"], list) or not claim["evidence_ids"]:
            return None, "claim.evidence_ids must be a non-empty array"
        if not all(isinstance(evidence_id, str) for evidence_id in claim["evidence_ids"]):
            return None, "claim.evidence_ids must contain strings"
        unknown = sorted(set(claim["evidence_ids"]) - evidence_ids)
        if unknown:
            return None, f"claim cites unknown evidence IDs: {unknown}"
        if not isinstance(claim["grounding"], str) or claim["grounding"] not in LABELS:
            return None, "claim.grounding is invalid"
    return parsed, None


def verify_answer_grounding(
    answer: str,
    evidence: list[dict[str, str]],
    *,
    client: Any,
    model: str = "",
) -> dict[str, Any]:
    """Verify an answer against supplied evidence without rewriting it."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a non-empty list")

    evidence_ids = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise ValueError("each evidence item must contain exactly id and text")
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValueError("evidence.id must be a non-empty string")
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("evidence.text must be a non-empty string")
        if item["id"] in evidence_ids:
            raise ValueError(f"duplicate evidence id: {item['id']}")
        evidence_ids.add(item["id"])

    payload = {
        "model": model,
        "instructions": PROMPT,
        "input": json.dumps(
            {"answer": answer, "evidence": evidence},
            ensure_ascii=False,
        ),
    }
    try:
        response = client.create(payload)
    except Exception as exc:
        return {
            "status": "error",
            "assessment": None,
            "error_type": "provider_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    parsed, parse_error = parse_grounding_response(
        _response_text(response), evidence_ids,
    )
    if parse_error:
        return {
            "status": "error",
            "assessment": None,
            "error_type": "malformed_response",
            "error": parse_error,
        }

    claims = parsed["claims"]
    return {
        "status": "ok",
        "assessment": {
            "answer": answer,
            "claims": claims,
            "fully_grounded": bool(claims) and all(
                claim["grounding"] == "supported" for claim in claims
            ),
        },
        "error_type": None,
        "error": "",
    }


__all__ = ["parse_grounding_response", "verify_answer_grounding"]
