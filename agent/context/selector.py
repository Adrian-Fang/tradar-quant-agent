"""Deterministic context selection for already-available context items."""

from __future__ import annotations

from typing import Any


PRIORITY = {
    "current_instruction": 5,
    "task_state": 4,
    "runtime_truth": 4,
    "user_preference": 3,
    "retrieved_knowledge": 2,
    "history": 1,
}


def select_context(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    selected = []
    dropped = []

    for item in items:
        kind = item["kind"]
        if kind not in PRIORITY:
            raise ValueError(f"unsupported selectable context kind: {kind}")
        if item.get("relevant", True) is False:
            dropped.append({"id": item["id"], "reason": "irrelevant"})
            continue

        key = item.get("key")
        higher_conflict = key is not None and any(
            other is not item
            and other.get("relevant", True) is not False
            and other.get("stale", False) is not True
            and other.get("key") == key
            and other["kind"] in PRIORITY
            and PRIORITY[other["kind"]] > PRIORITY[kind]
            for other in items
        )
        if higher_conflict:
            dropped.append({"id": item["id"], "reason": "superseded"})
        else:
            selected.append(item)

    return {"selected": selected, "dropped": dropped}


__all__ = ["PRIORITY", "select_context"]
