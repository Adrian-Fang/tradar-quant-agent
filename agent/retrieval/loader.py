"""Minimal loader for Markdown research records."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPO_ROOT / "resources" / "knowledge" / "research"


def _parse_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
        if not value:
            return []
        return [part.strip().strip("'\"") for part in value.split(",")]
    return value.strip("'\"")


def _parse_record(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"missing frontmatter: {path}")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"unterminated frontmatter: {path}") from exc

    metadata = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_value(value)

    body_lines = lines[end + 1 :]
    sections = {}
    current_section = None
    for line in body_lines:
        if line.startswith("# "):
            current_section = line[2:].strip()
            sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line)

    text = "\n".join(body_lines).strip()
    relative_path = str(path.relative_to(REPO_ROOT))
    return {
        "research_id": metadata["research_id"],
        "path": relative_path,
        "metadata": metadata,
        "title": metadata.get("topic", path.stem),
        "question": "\n".join(sections.get("Research Question", [])).strip(),
        "tags": metadata.get("tags", []),
        "text": text,
        "source": metadata.get("source_type", ""),
        "source_ref": metadata.get("source_ref", []),
        "provenance": "\n".join(sections.get("Provenance", [])).strip(),
    }


def load_research_records(
    root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    research_root = root or RESEARCH_ROOT
    return tuple(
        _parse_record(path)
        for path in sorted(research_root.glob("*.md"))
    )


__all__ = ["REPO_ROOT", "RESEARCH_ROOT", "load_research_records"]
