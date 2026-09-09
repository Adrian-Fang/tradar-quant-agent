"""Deterministic budget handling for already-selected context items."""

from __future__ import annotations

from typing import Any

from .selection import PRIORITY


ESSENTIAL_KINDS = {"current_instruction", "task_state", "runtime_truth"}


def compact_context(items: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    """Keep, losslessly compact, or drop selected context to fit a character budget."""
    if budget < 0:
        raise ValueError("budget must be non-negative")

    original_size = sum(len(item["text"]) for item in items)
    if original_size <= budget:
        return {"status": "ok", "kept": list(items), "compacted": [], "dropped": []}

    essential_size = sum(
        len(item["text"]) for item in items if item["kind"] in ESSENTIAL_KINDS
    )
    if essential_size > budget:
        return {
            "status": "insufficient_budget",
            "kept": [item for item in items if item["kind"] in ESSENTIAL_KINDS],
            "compacted": [],
            "dropped": [
                {"id": item["id"], "reason": "budget"}
                for item in items
                if item["kind"] not in ESSENTIAL_KINDS
            ],
        }

    entries = [{"status": "kept", "item": item} for item in items]
    size = original_size
    ordered_indexes = sorted(
        range(len(items)), key=lambda index: (PRIORITY[items[index]["kind"]], index)
    )

    for index in ordered_indexes:
        if size <= budget:
            break
        item = items[index]
        compacted_lines = []
        seen_structured_lines = set()
        previous_blank = False
        for raw_line in item["text"].splitlines():
            line = " ".join(raw_line.split())
            if not line:
                if not previous_blank:
                    compacted_lines.append("")
                previous_blank = True
                continue
            if "=" in line and line in seen_structured_lines:
                continue
            if "=" in line:
                seen_structured_lines.add(line)
            compacted_lines.append(line)
            previous_blank = False

        compacted_text = "\n".join(compacted_lines).strip()
        if compacted_text != item["text"]:
            compacted_item = dict(item)
            compacted_item["text"] = compacted_text
            entries[index] = {"status": "compacted", "item": compacted_item}
            size += len(compacted_text) - len(item["text"])

    for index in ordered_indexes:
        if size <= budget:
            break
        item = items[index]
        if item["kind"] in ESSENTIAL_KINDS:
            continue
        size -= len(entries[index]["item"]["text"])
        entries[index] = {"status": "dropped", "item": item}

    return {
        "status": "ok",
        "kept": [entry["item"] for entry in entries if entry["status"] == "kept"],
        "compacted": [entry["item"] for entry in entries if entry["status"] == "compacted"],
        "dropped": [
            {"id": entry["item"]["id"], "reason": "budget"}
            for entry in entries
            if entry["status"] == "dropped"
        ],
    }


__all__ = ["compact_context"]
