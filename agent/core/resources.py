"""Small loaders for non-executable repository resources."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = REPO_ROOT / "resources"


def load_jsonl(relative_path: str) -> tuple[dict[str, Any], ...]:
    path = RESOURCE_ROOT / relative_path
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL item at {path}:{line_number} must be an object")
            cases.append(value)
    return tuple(cases)


def load_prompt(relative_path: str) -> str:
    return (RESOURCE_ROOT / relative_path).read_text(encoding="utf-8").strip()


def expand_tokens(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        return value
    if isinstance(value, Mapping):
        return {key: expand_tokens(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_tokens(item, replacements) for item in value]
    return value


__all__ = ["REPO_ROOT", "RESOURCE_ROOT", "expand_tokens", "load_jsonl", "load_prompt"]
