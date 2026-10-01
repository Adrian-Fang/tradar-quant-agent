"""Canonical research-record contract (v1), with an explicit unversioned adapter."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any
import re
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPO_ROOT / "resources" / "knowledge" / "research"
SCHEMA_VERSION = "1.0"
SECTIONS = ("Research Question", "Method", "Key Findings", "Conclusion", "Caveats", "Provenance")
STATUSES = {"exploratory", "promising", "validated", "rejected", "inconclusive", "superseded"}
METADATA_FIELDS = {"schema_version", "research_id", "market", "topic", "strategy", "status",
                   "source_type", "source_task_id", "date", "tags", "source_ref", "artifacts",
                   "supersedes", "title"}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"invalid/duplicate frontmatter key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _parse_record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"missing frontmatter: {path}")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"unterminated frontmatter: {path}") from exc

    metadata = yaml.load("\n".join(lines[1:end]), Loader=_UniqueLoader)
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be a mapping")
    if unknown := metadata.keys() - METADATA_FIELDS:
        raise ValueError(f"unsupported frontmatter fields: {sorted(unknown)}")
    legacy = "schema_version" not in metadata
    if not legacy and metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {metadata['schema_version']!r}")
    metadata["schema_version"] = SCHEMA_VERSION
    # Public records need no internal task ID or private source reference.
    metadata.setdefault("source_task_id", None)
    metadata.setdefault("source_ref", [])
    for key in ("research_id", "market", "topic", "strategy", "status", "source_type"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if "/" in metadata["research_id"] or "\n" in metadata["research_id"]:
        raise ValueError("research_id cannot contain slash/newline")
    if metadata["status"] not in STATUSES:
        raise ValueError("invalid status")
    if metadata["source_task_id"] is not None:
        if not isinstance(metadata.get("source_task_id"), str) or not metadata["source_task_id"].strip():
            raise ValueError("source_task_id must be a nonempty string")
    value = metadata.get("date")
    if type(value) is date:
        metadata["date"] = value.isoformat()
    elif isinstance(value, str):
        try:
            metadata["date"] = date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError("date must be an ISO date") from exc
    else:
        raise ValueError("date must be an ISO date")
    for key in ("tags", "source_ref", "artifacts", "supersedes"):
        value = metadata.get(key)
        if key == "source_ref" and isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{key} must be an array of nonempty strings")
        metadata[key] = value
    if "title" in metadata and (not isinstance(metadata["title"], str) or not metadata["title"].strip()):
        raise ValueError("title must be a nonempty string")

    sections, current, fence = {}, None, None
    for line in lines[end + 1:]:
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})(.*)$", stripped) if len(line) - len(stripped) <= 3 else None
        if marker:
            run, suffix = marker.groups()
            if fence is None and (run[0] != "`" or "`" not in suffix):
                fence = run
            elif fence and run[0] == fence[0] and len(run) >= len(fence) and not suffix.strip():
                fence = None
        if fence is None and line.startswith("# "):
            current = line[2:].strip()
            if current not in SECTIONS or current in sections:
                raise ValueError(f"unknown/duplicate required section: {current}")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
        elif line.strip():
            raise ValueError("content before Research Question section")
    if fence is not None:
        raise ValueError("unterminated fenced block")
    sections = {key: "\n".join(value).strip() for key, value in sections.items()}
    for key in SECTIONS:
        if not sections.get(key):
            raise ValueError(f"missing/empty required section: {key}")
    relative_path = path.resolve().relative_to(REPO_ROOT).as_posix() if path.resolve().is_relative_to(REPO_ROOT) else path.name
    return {
        "research_id": metadata["research_id"],
        "path": relative_path,
        "metadata": metadata,
        "title": metadata.get("title", metadata["topic"]),
        "question": sections["Research Question"],
        "tags": metadata["tags"],
        "text": "\n".join(lines[end + 1:]).strip(), "sections": sections,
        "source": metadata["source_type"], "source_ref": metadata["source_ref"],
        "provenance": sections["Provenance"], "source_hash": sha256(raw).hexdigest(),
        "schema_version": SCHEMA_VERSION, "legacy_adapter": legacy,
    }


def load_research_records(
    root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    research_root = Path(root or RESEARCH_ROOT)
    if not research_root.is_dir():
        raise ValueError(f"canonical corpus directory unavailable: {research_root}")
    records, seen = [], set()
    for path in sorted(research_root.glob("*.md")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"canonical record must be a regular non-symlink file: {path}")
        try:
            record = _parse_record(path)
        except (ValueError, KeyError, yaml.YAMLError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if record["research_id"] in seen:
            raise ValueError(f"duplicate research_id: {record['research_id']}")
        seen.add(record["research_id"])
        records.append(record)
    # References to deleted historical records are valid; cycles among live records are not.
    graph = {record["research_id"]: record["metadata"]["supersedes"] for record in records}
    active, done = set(), set()

    def visit(research_id):
        if research_id in active:
            raise ValueError(f"supersession cycle at research_id: {research_id}")
        if research_id in done or research_id not in graph:
            return
        active.add(research_id)
        for previous in graph[research_id]:
            visit(previous)
        active.remove(research_id)
        done.add(research_id)

    for research_id in graph:
        visit(research_id)
    return tuple(records)


__all__ = ["REPO_ROOT", "RESEARCH_ROOT", "load_research_records"]
