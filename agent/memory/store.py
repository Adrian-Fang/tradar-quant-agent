"""Append-only lifecycle for atomic memory records."""

from __future__ import annotations

from typing import Any


def active_memories(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    superseded = {
        record["supersedes_id"]
        for record in records
        if record.get("supersedes_id") is not None
    }
    return tuple(record for record in records if record["id"] not in superseded)


def apply_memory_decision(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    decision: dict[str, Any],
    *,
    new_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    current = tuple(dict(record) for record in records)
    if not isinstance(decision, dict) or decision.get("action") not in {"write", "update", "ignore"}:
        raise ValueError("decision.action must be write, update, or ignore")

    action = decision["action"]
    if action == "ignore":
        return current
    if not isinstance(new_id, str) or not new_id.strip():
        raise ValueError("new_id is required for write/update")
    if any(record["id"] == new_id for record in current):
        raise ValueError(f"duplicate memory id: {new_id}")

    text = decision.get("memory")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("memory must be a non-empty string for write/update")

    supersedes_id = None
    if action == "update":
        supersedes_id = decision.get("supersedes_id")
        if not isinstance(supersedes_id, str) or not supersedes_id.strip():
            raise ValueError("update requires supersedes_id")
        if supersedes_id not in {record["id"] for record in current}:
            raise ValueError("update supersedes a missing memory")
        if supersedes_id not in {record["id"] for record in active_memories(current)}:
            raise ValueError("update can only supersede an active memory")
    elif decision.get("supersedes_id") is not None:
        raise ValueError("write must not supersede a memory")

    return current + ({
        "id": new_id,
        "text": text,
        "supersedes_id": supersedes_id,
    },)
