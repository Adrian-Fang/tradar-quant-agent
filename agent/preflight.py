"""One-call entry preflight; research evidence remains the existing pipeline's job."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .core.providers import provider_error_type
from .core.resources import load_prompt
from .tools.calling import _extract_function_calls


PROMPT = load_prompt("prompts/preflight.md")
OUTCOMES = {"direct", "needs_input", "research"}
TOOL_NAME = "submit_preflight"
TOOL_SCHEMA = {
    "type": "function", "name": TOOL_NAME,
    "description": "Answer directly, request minimal clarification, or enter research.",
    "parameters": {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": sorted(OUTCOMES)},
            "answer": {"type": "string"},
        },
        "required": ["outcome", "answer"], "additionalProperties": False,
    },
    "strict": True,
}


def build_preflight_payload(
    user_request: str, *, model: str = "", capabilities: list[str] | None = None,
    product_boundaries: list[str] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "model": model, "instructions": PROMPT,
        "input": json.dumps({
            "user_request": user_request,
            "runtime_metadata": {"product": "Tradar", "model": model or None,
                                 "capabilities": capabilities or [],
                                 "product_boundaries": product_boundaries or []},
            "conversation_context": conversation_history or [],
        }, ensure_ascii=False),
        "tools": [TOOL_SCHEMA], "tool_choice": "required", "parallel_tool_calls": False,
    }


def parse_preflight_response(response: Any) -> tuple[dict[str, str] | None, str | None]:
    try:
        calls = _extract_function_calls(response)
        if calls:
            if len(calls) != 1 or calls[0]["name"] != TOOL_NAME:
                return None, "preflight response must contain one submit_preflight call"
            parsed = calls[0]["arguments"]
        else:
            text = response.get("output_text", "") if isinstance(response, Mapping) else getattr(response, "output_text", "")
            parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        return None, f"preflight response is not valid JSON: {exc}"
    if not isinstance(parsed, dict) or set(parsed) != {"outcome", "answer"}:
        return None, "preflight response must contain exactly outcome and answer"
    if not isinstance(parsed["outcome"], str) or parsed["outcome"] not in OUTCOMES:
        return None, "preflight.outcome is invalid"
    if not isinstance(parsed["answer"], str):
        return None, "preflight.answer must be a string"
    if parsed["outcome"] == "research":
        if parsed["answer"]:
            return None, "research preflight must not contain an answer"
    elif not parsed["answer"].strip():
        return None, "direct/needs_input preflight must contain a non-empty answer"
    return parsed, None


def run_preflight(user_request: str, *, client: Any, **options: Any) -> dict[str, Any]:
    try:
        response = client.create(build_preflight_payload(user_request, **options))
    except Exception as exc:
        return {"status": "error", "result": None, "error_type": provider_error_type(exc),
                "error": f"{type(exc).__name__}: {exc}"}
    parsed, error = parse_preflight_response(response)
    return {"status": "error" if error else "ok", "result": parsed,
            "error_type": "malformed_response" if error else None, "error": error or ""}
