"""Small loaders for non-executable repository resources."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = REPO_ROOT / "resources"


def load_json(relative_path: str) -> tuple[dict[str, Any], ...]:
    value = json.loads((RESOURCE_ROOT / relative_path).read_text(encoding="utf-8"))
    return tuple(value["cases"])


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


__all__ = ["REPO_ROOT", "RESOURCE_ROOT", "expand_tokens", "load_json", "load_prompt"]
