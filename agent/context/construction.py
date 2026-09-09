"""Deterministic assembly of the final provider context."""

from __future__ import annotations

import json
from typing import Any

from .selection import PRIORITY


def construct_context(
    user_request: str,
    compaction_result: dict[str, Any],
    *,
    system_rules: str = "",
    product_rules: str = "",
) -> dict[str, Any]:
    if compaction_result["status"] != "ok":
        return {
            "status": compaction_result["status"],
            "instructions": None,
            "input": None,
            "context": None,
        }

    context = sorted(
        compaction_result["context"],
        key=lambda item: -PRIORITY[item["kind"]],
    )
    instruction_sections = []
    if system_rules:
        instruction_sections.append(f"System Rules:\n{system_rules}")
    if product_rules:
        instruction_sections.append(f"Product Rules:\n{product_rules}")

    instructions = "\n\n".join(instruction_sections)
    input_text = (
        f"User Request:\n{user_request}\n\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    return {
        "status": "ok",
        "instructions": instructions,
        "input": input_text,
        "context": context,
    }


__all__ = ["construct_context"]
