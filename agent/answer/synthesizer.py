"""Provider-agnostic answer synthesis from supplied evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.resources import load_prompt


PROMPT = load_prompt("prompts/answer_synthesis.md")
STATUSES = {"success", "insufficient_evidence"}
SYNTHESIS_TOOL_NAME = "submit_synthesized_answer"
SYNTHESIS_TOOL_SCHEMA = {
    "type": "function",
    "name": SYNTHESIS_TOOL_NAME,
    "description": "Submit the final answer and the supplied evidence IDs it uses.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": sorted(STATUSES)},
            "answer": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status", "answer", "evidence_ids"],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_synthesis_payload(
    user_request: str,
    evidence: list[dict[str, str]],
    *,
    model: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    input_data = {
        "user_request": user_request,
        "evidence": evidence,
    }
    if conversation_history is not None:
        input_data["conversation_context"] = conversation_history
    return {
        "model": model,
        "instructions": PROMPT,
        "input": json.dumps(input_data, ensure_ascii=False),
        "tools": [SYNTHESIS_TOOL_SCHEMA],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }


def response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def response_payload(response: Any) -> Any:
    """Read structured function arguments, with JSON text as a fallback."""
    def field(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(
            value, name, default
        )

    calls = [
        item
        for item in field(response, "output", []) or []
        if field(item, "type") == "function_call"
    ]
    if calls:
        if len(calls) != 1 or field(calls[0], "name") != SYNTHESIS_TOOL_NAME:
            return {"_invalid_tool_call": True}
        arguments = field(calls[0], "arguments")
        if isinstance(arguments, Mapping):
            return dict(arguments)
        return arguments if isinstance(arguments, str) else {"_invalid_arguments": True}
    return response_text(response)


def parse_synthesis_response(
    response_value: Any,
    evidence_ids: set[str],
) -> tuple[Any | None, str | None]:
    if isinstance(response_value, str):
        try:
            parsed = json.loads(response_value.strip())
        except json.JSONDecodeError as exc:
            return None, f"response is not valid JSON: {exc}"
    else:
        parsed = response_value
    if not isinstance(parsed, dict):
        return parsed, "response JSON must be an object"
    if set(parsed) != {"status", "answer", "evidence_ids"}:
        return parsed, "response must contain exactly status, answer, and evidence_ids"
    if not isinstance(parsed["status"], str) or parsed["status"] not in STATUSES:
        return parsed, "response.status is invalid"
    if not isinstance(parsed["answer"], str) or not parsed["answer"].strip():
        return parsed, "response.answer must be a non-empty string"
    if not isinstance(parsed["evidence_ids"], list):
        return parsed, "response.evidence_ids must be an array"
    if not all(isinstance(value, str) and value for value in parsed["evidence_ids"]):
        return parsed, "response.evidence_ids must contain non-empty strings"
    if len(parsed["evidence_ids"]) != len(set(parsed["evidence_ids"])):
        return parsed, "response.evidence_ids must not contain duplicates"
    unknown = sorted(set(parsed["evidence_ids"]) - evidence_ids)
    if unknown:
        return parsed, f"response cites unknown evidence IDs: {unknown}"
    if parsed["status"] == "success" and not parsed["evidence_ids"]:
        return parsed, "success response must cite at least one evidence ID"
    return parsed, None


def synthesize_answer(
    user_request: str,
    evidence: list[dict[str, str]],
    *,
    client: Any,
    model: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("user_request must be a non-empty string")
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

    try:
        response = client.create(build_synthesis_payload(
            user_request,
            evidence,
            model=model,
            conversation_history=conversation_history,
        ))
    except Exception as exc:
        return {
            "status": "error",
            "result": None,
            "error_type": "provider_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    parsed, parse_error = parse_synthesis_response(response_payload(response), evidence_ids)
    if parse_error:
        return {
            "status": "error",
            "result": None,
            "error_type": "malformed_response",
            "error": parse_error,
        }
    return {
        "status": "ok",
        "result": parsed,
        "error_type": None,
        "error": "",
    }


__all__ = [
    "PROMPT",
    "STATUSES",
    "SYNTHESIS_TOOL_NAME",
    "SYNTHESIS_TOOL_SCHEMA",
    "build_synthesis_payload",
    "parse_synthesis_response",
    "response_payload",
    "response_text",
    "synthesize_answer",
]
