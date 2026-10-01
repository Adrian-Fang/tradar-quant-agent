"""Section-aware chunks; exact section content and tables are never sliced."""

from hashlib import sha256
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .loader import REPO_ROOT, SCHEMA_VERSION

GROUPS = {
    "question_method": ("Research Question", "Method"),
    "findings": ("Key Findings",),
    "conclusion_caveats": ("Conclusion", "Caveats"),
}


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def load_profile(path: Path | None = None) -> dict:
    profile = json.loads((path or REPO_ROOT / "resources/retrieval/profile.json").read_text())
    if profile["schema_version"] != SCHEMA_VERSION or profile["chunker_version"] != "sections-v1":
        raise ValueError("unsupported research schema/chunker profile")
    if profile["dense"]["size"] != 1024 or profile["dense"]["distance"] != "Cosine":
        raise ValueError("AE-14 requires 1024-dimensional cosine vectors")
    if profile["dense"]["name"] != "dense" or profile["sparse"] != {"name": "bm25", "model": "qdrant/bm25", "modifier": "idf"}:
        raise ValueError("unsupported named-vector profile")
    return profile


def profile_hash(profile: dict) -> str:
    return digest(json.dumps(profile, sort_keys=True, separators=(",", ":")))


def chunk_record(record: dict, profile: dict) -> list[dict]:
    result = []
    for group, sections in GROUPS.items():
        text = "\n\n".join(f"# {name}\n\n{record['sections'][name]}" for name in sections)
        chunk_id = f"{record['research_id']}/{group}/1"
        embedding_input = f"Research ID: {record['research_id']}\nTitle: {record['title']}\nTags: {', '.join(record['tags'])}\n\n{text}"
        # ponytail: preserve whole semantic blocks; reject overlong blocks rather than silently truncate.
        if len(embedding_input.encode("utf-8")) > 24000:
            raise ValueError(f"{chunk_id}: semantic block too long; explicit subdivision required")
        result.append({
            "chunk_id": chunk_id, "point_id": str(uuid5(NAMESPACE_URL, f"tradar/{profile_hash(profile)}/{chunk_id}")),
            "section_group": group, "sections": list(sections), "part_number": 1,
            "text": text, "embedding_input": embedding_input,
            "chunk_hash": digest(text), "embedding_input_hash": digest(embedding_input),
        })
    return result
