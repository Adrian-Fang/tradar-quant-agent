"""Provider-agnostic runtime HITL approval gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.resources import load_prompt


PROMPT = load_prompt("prompts/hitl.md")
DECISIONS = ("proceed", "needs_approval", "blocked")


def build_gate_payload(
    user_request: str,
    proposed_action: dict[str, Any],
    existing_approval: dict[str, Any],
    product_boundaries: list[str],
    *,
    model: str = "",
) -> dict[str, str]:
    return {
        "model": model,
        "instructions": PROMPT,
        "input": json.dumps({
            "user_request": user_request,
            "proposed_action": proposed_action,
            "existing_approval": existing_approval,
            "product_boundaries": product_boundaries,
        }, ensure_ascii=False),
    }


def response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def parse_hitl_response(text: str) -> tuple[Any | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return parsed, "response JSON must be an object"
    if set(parsed) != {"decision", "approval_request", "reason"}:
        return parsed, "response must contain exactly decision, approval_request, and reason"
    if parsed["decision"] not in DECISIONS:
        return parsed, "response.decision is invalid"
    if parsed["approval_request"] is not None and not isinstance(parsed["approval_request"], str):
        return parsed, "response.approval_request must be a string or null"
    if parsed["decision"] == "needs_approval" and (
        not isinstance(parsed["approval_request"], str)
        or not parsed["approval_request"].strip()
    ):
        return parsed, "needs_approval requires a non-empty approval_request"
    if parsed["decision"] != "needs_approval" and parsed["approval_request"] is not None:
        return parsed, "proceed and blocked require a null approval_request"
    if not isinstance(parsed["reason"], str):
        return parsed, "response.reason must be a string"
    return parsed, None


def gate_action(
    user_request: str,
    proposed_action: dict[str, Any],
    existing_approval: dict[str, Any],
    product_boundaries: list[str],
    *,
    client: Any,
    model: str = "",
) -> dict[str, Any]:
    try:
        response = client.create(build_gate_payload(
            user_request,
            proposed_action,
            existing_approval,
            product_boundaries,
            model=model,
        ))
    except Exception as exc:
        return {
            "status": "error",
            "decision": None,
            "approval_request": None,
            "reason": "",
            "error_type": "provider_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    parsed, error = parse_hitl_response(response_text(response))
    if error:
        return {
            "status": "error",
            "decision": None,
            "approval_request": None,
            "reason": "",
            "error_type": "malformed_response",
            "error": error,
        }
    return {
        "status": "ok",
        "decision": parsed["decision"],
        "approval_request": parsed["approval_request"],
        "reason": parsed["reason"],
        "error_type": None,
        "error": "",
    }


__all__ = [
    "DECISIONS",
    "PROMPT",
    "build_gate_payload",
    "gate_action",
    "parse_hitl_response",
    "response_text",
]
