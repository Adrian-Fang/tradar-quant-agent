"""Runtime memory recall over the active append-only memory view."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.resources import load_prompt
from .store import active_memories


RECALL_PROMPT = load_prompt("prompts/memory_recall.md")


def build_recall_payload(
    user_request: str,
    memories: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    model: str = "",
) -> dict[str, str]:
    return {
        "model": model,
        "instructions": RECALL_PROMPT,
        "input": json.dumps({
            "user_request": user_request,
            "active_memories": list(memories),
        }, ensure_ascii=False),
    }


def response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def parse_recall_response(
    text: str,
    memory_ids: list[str],
) -> tuple[Any | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return parsed, "response JSON must be an object"
    if set(parsed) != {"selected_ids", "reason"}:
        return parsed, "response must contain exactly selected_ids and reason"
    selected_ids = parsed["selected_ids"]
    if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
        return parsed, "response.selected_ids must be a list of strings"
    if len(selected_ids) != len(set(selected_ids)):
        return parsed, "response.selected_ids must not contain duplicates"
    known_ids = set(memory_ids)
    if any(memory_id not in known_ids for memory_id in selected_ids):
        return parsed, "response.selected_ids contains an unknown memory ID"
    if selected_ids != [memory_id for memory_id in memory_ids if memory_id in selected_ids]:
        return parsed, "response.selected_ids must preserve input memory order"
    if not isinstance(parsed["reason"], str):
        return parsed, "response.reason must be a string"
    return parsed, None


def recall_memories(
    user_request: str,
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    client: Any,
    model: str = "",
) -> dict[str, Any]:
    active = active_memories(records)
    if not active:
        return {
            "status": "ok",
            "selected_ids": [],
            "selected_memories": [],
            "error_type": None,
            "error": "",
        }

    memory_ids = [memory["id"] for memory in active]
    try:
        response = client.create(build_recall_payload(user_request, active, model=model))
    except Exception as exc:
        return {
            "status": "error",
            "selected_ids": [],
            "selected_memories": [],
            "error_type": "provider_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    parsed, error = parse_recall_response(response_text(response), memory_ids)
    if error:
        return {
            "status": "error",
            "selected_ids": [],
            "selected_memories": [],
            "error_type": "malformed_response",
            "error": error,
        }

    selected_ids = parsed["selected_ids"]
    return {
        "status": "ok",
        "selected_ids": selected_ids,
        "selected_memories": [
            memory for memory in active if memory["id"] in selected_ids
        ],
        "error_type": None,
        "error": "",
    }


__all__ = [
    "RECALL_PROMPT",
    "build_recall_payload",
    "parse_recall_response",
    "recall_memories",
    "response_text",
]
