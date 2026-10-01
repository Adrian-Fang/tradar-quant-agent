"""Explicit, incremental indexing. Markdown is authoritative; state is rebuildable."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
from uuid import UUID, uuid4

from .chunking import chunk_record, digest, load_profile, profile_hash
from .embeddings import OllamaDense, validate_vectors
from .loader import REPO_ROOT, RESEARCH_ROOT, load_research_records
from .qdrant_store import QdrantStore, canonical_filter_payload, query_spec

STATE_PATH = REPO_ROOT / ".runtime/knowledge_index/manifest.json"


@contextmanager
def index_lock(path: Path, *, exclusive: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    # ponytail: one local index writer; distributed ingestion is outside AE-14.
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    try:
        if not isinstance(value, dict) or type(value.get("state_version")) is not int or value["state_version"] != 1 or not isinstance(value["records"], dict):
            raise ValueError("invalid state version/records")
        if not isinstance(value["ready"], bool) or value["profile_hash"] != profile_hash(value["profile"]):
            raise ValueError("invalid readiness/profile hash")
        prefix = f"tradar_research_{value['profile_hash'][:16]}"
        collection = value["collection"]
        if collection != prefix and (not isinstance(collection, str) or not collection.startswith(prefix + "_") or len(collection[len(prefix) + 1:]) != 12 or any(char not in "0123456789abcdef" for char in collection[len(prefix) + 1:])):
            raise ValueError("collection is outside the index profile namespace")
        if not isinstance(value["corpus_root"], str) or not isinstance(value["known_point_ids"], list):
            raise ValueError("invalid corpus root/point journal")
        for point_id in value["known_point_ids"]:
            UUID(point_id)
        for record in value["records"].values():
            if not isinstance(record["path"], str) or not isinstance(record["source_hash"], str) or len(record["source_hash"]) != 64 or not isinstance(record["chunks"], dict):
                raise ValueError("invalid record identity/hash/chunks")
            for chunk in record["chunks"].values():
                UUID(chunk["point_id"])
                if chunk["point_id"] not in value["known_point_ids"]:
                    raise ValueError("point missing from journal")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"invalid knowledge-index manifest; explicit rebuild required: {exc}") from exc
    return value


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolved_profile(profile: dict, revision: str) -> dict:
    if not isinstance(revision, str) or not revision or revision == "ollama-digest-at-index-time":
        raise ValueError("actual embedding model revision is required")
    result = deepcopy(profile)
    result["dense"]["revision"] = revision
    return result


def repo_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _records_state(records: tuple, chunks: dict) -> dict:
    superseded = {value for record in records for value in record["metadata"]["supersedes"]}
    return {record["research_id"]: {
        "source_hash": record["source_hash"], "path": record["path"],
        "is_superseded": record["metadata"]["status"] == "superseded" or record["research_id"] in superseded,
        "chunks": {chunk["chunk_id"]: {key: chunk[key] for key in ("point_id", "chunk_hash", "embedding_input_hash")} for chunk in chunks[record["research_id"]]},
    } for record in records}


def _validate_collection(store, collection: str, expected: dict, profile: dict, records: tuple) -> None:
    ids = [chunk["point_id"] for record in expected.values() for chunk in record["chunks"].values()]
    if store.count(collection) != len(ids):
        raise RuntimeError("index validation: point count mismatch")
    found = {str(point["id"]): point for point in store.points(collection, ids)}
    canonical = {record["research_id"]: record for record in records}
    for research_id, record in expected.items():
        for chunk_id, chunk in record["chunks"].items():
            point = found.get(chunk["point_id"], {})
            payload, vectors = point.get("payload", {}), point.get("vector", {})
            metadata = canonical_filter_payload(canonical[research_id], is_superseded=record["is_superseded"])
            if any(type(payload.get(key)) is not type(value) or payload[key] != value for key, value in metadata.items()) or payload.get("source_path") != record["path"]:
                raise RuntimeError(f"index validation: filter metadata mismatch: {chunk_id}")
            if payload.get("source_hash") != record["source_hash"] or payload.get("chunk_id") != chunk_id or payload.get("research_id") != research_id:
                raise RuntimeError(f"index validation: identity/source mismatch: {chunk_id}")
            if any(payload.get(key) != chunk[key] for key in ("chunk_hash", "embedding_input_hash")) or payload.get("profile_hash") != profile_hash(profile):
                raise RuntimeError(f"index validation: hash/profile mismatch: {chunk_id}")
            validate_vectors([vectors.get("dense")], 1)
            if not isinstance(payload.get("text"), str) or digest(payload["text"]) != chunk["chunk_hash"]:
                raise RuntimeError(f"index validation: text/hash mismatch: {chunk_id}")
            sparse = vectors.get("bm25")
            if (not isinstance(sparse, dict) or not isinstance(sparse.get("indices"), list) or not sparse["indices"]
                    or not isinstance(sparse.get("values"), list) or len(sparse["indices"]) != len(sparse["values"])
                    or any(type(index) is not int or index < 0 for index in sparse["indices"])
                    or sparse["indices"] != sorted(set(sparse["indices"]))
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in sparse["values"])):
                raise RuntimeError(f"index validation: native BM25 vector missing/invalid: {chunk_id}")
    # Prove each branch, not just acceptance of a fused request. Include superseded
    # records for this health check even when the whole corpus is historical.
    research_id = next(iter(expected))
    probe_ids = {chunk["point_id"] for chunk in expected[research_id]["chunks"].values()}
    probe_id = sorted(probe_ids)[0]
    vector = found[probe_id]["vector"]["dense"]
    for mode in ("dense", "bm25", "hybrid"):
        results = store.query(collection, query_spec(f"Research {research_id}", vector, profile, mode,
                                                    {"research_id": research_id, "include_superseded": True}))
        result_ids = {str(point.get("id")) for point in results}
        if probe_id not in result_ids or not result_ids <= probe_ids:
            raise RuntimeError(f"index validation: {mode} retrieval proof failed")


def sync(*, store, embedder, profile: dict | None = None, root: Path = RESEARCH_ROOT,
         state_path: Path = STATE_PATH, rebuild: bool = False) -> dict:
    with index_lock(state_path, exclusive=True):
        # Complete validation before interpreting missing files as deletions.
        records = load_research_records(root)
        if not records:
            raise ValueError("empty canonical corpus: publication requires a real retrieval probe; no index/alias mutations performed")
        profile = resolved_profile(profile or load_profile(), embedder.revision)
        chunks = {record["research_id"]: chunk_record(record, profile) for record in records}
        expected = _records_state(records, chunks)
        try:
            old = read_manifest(state_path)
        except (ValueError, json.JSONDecodeError):
            if not rebuild:
                raise
            old = None
        if old and old.get("corpus_root") != str(root.resolve()):
            raise ValueError("manifest belongs to another corpus directory")
        if old and old["profile_hash"] != profile_hash(profile) and not rebuild:
            raise ValueError("index profile/model revision changed; run rebuild")
        if not old and not rebuild and store.alias_target(profile["alias"]):
            raise RuntimeError("index alias exists without local state; run rebuild")
        if old and old.get("ready") and not rebuild and old["records"] == expected:
            if not store.exists(old["collection"]) or store.alias_target(profile["alias"]) != old["collection"]:
                raise RuntimeError("index collection/alias missing or changed; run rebuild")
            return {"status": "unchanged", "records": len(records), "embedded": 0, "upserted": 0, "deleted": 0, "collection": old["collection"]}
        if old and old.get("ready") and not rebuild and store.alias_target(profile["alias"]) != old["collection"]:
            raise RuntimeError("index alias changed externally; explicit rebuild required")
        collection = (f"tradar_research_{profile_hash(profile)[:16]}_{uuid4().hex[:12]}" if rebuild else
                      old["collection"] if old else f"tradar_research_{profile_hash(profile)[:16]}")
        if not old and not rebuild and store.exists(collection):
            raise RuntimeError("index collection exists without local state; run rebuild")
        if old and old.get("ready") and not rebuild and not store.exists(collection):
            raise RuntimeError("index collection missing; run rebuild")
        previous = {} if rebuild or not old or not old.get("ready") else old["records"]
        to_write, to_embed, reusable = [], [], {}
        for record in records:
            prior = previous.get(record["research_id"], {})
            if prior == expected[record["research_id"]]:
                continue
            for chunk in chunks[record["research_id"]]:
                to_write.append((record, chunk))
                if prior.get("chunks", {}).get(chunk["chunk_id"], {}).get("embedding_input_hash") == chunk["embedding_input_hash"]:
                    reusable[chunk["point_id"]] = None
                else:
                    to_embed.append(chunk)
        if reusable:
            for point in store.points(collection, list(reusable)):
                reusable[str(point["id"])] = point.get("vector", {}).get("dense")
            for vector in reusable.values():
                validate_vectors([vector], 1)
        vectors = validate_vectors(embedder([chunk["embedding_input"] for chunk in to_embed]), len(to_embed)) if to_embed else []
        if embedder.revision != profile["dense"]["revision"]:
            raise RuntimeError("embedding model revision changed during indexing; retry with rebuild")
        dense = {**reusable, **{chunk["point_id"]: vector for chunk, vector in zip(to_embed, vectors)}}
        superseded = {value for record in records for value in record["metadata"]["supersedes"]}
        timestamp, revision = datetime.now(timezone.utc).isoformat(), repo_revision()
        points = []
        for record, chunk in to_write:
            metadata = record["metadata"]
            payload = {
                **{key: metadata[key] for key in ("market", "topic", "status", "tags", "source_type", "source_ref", "source_task_id", "artifacts", "supersedes", "strategy")},
                **{key: chunk[key] for key in ("chunk_id", "section_group", "sections", "part_number", "text", "chunk_hash", "embedding_input_hash")},
                "research_id": record["research_id"], "source_path": record["path"], "title": record["title"],
                "date": f"{metadata['date']}T00:00:00Z", "source_hash": record["source_hash"],
                "is_superseded": metadata["status"] == "superseded" or record["research_id"] in superseded,
                "provenance": record["provenance"], "schema_version": profile["schema_version"],
                "chunker_version": profile["chunker_version"], "profile_version": profile["profile_version"],
                "profile_hash": profile_hash(profile), "embedding_model": profile["dense"]["model"],
                "embedding_revision": profile["dense"]["revision"], "indexed_at": timestamp, "repo_revision": revision,
            }
            points.append({"id": chunk["point_id"], "payload": payload, "vector": {"dense": dense[chunk["point_id"]], "bm25": {"text": chunk["embedding_input"], "model": "qdrant/bm25"}}})
        target_ids = {chunk["point_id"] for record in expected.values() for chunk in record["chunks"].values()}
        known = set(old.get("known_point_ids", [])) if old and not rebuild else set()
        delete_ids = sorted(known - target_ids)
        pending = {
            "state_version": 1, "ready": False, "collection": collection,
            "corpus_root": str(root.resolve()), "profile": profile, "profile_hash": profile_hash(profile),
            "records": expected, "known_point_ids": sorted(known | target_ids), "indexed_at": timestamp,
        }
        # Journal possible point IDs before any remote mutation, so interrupted sync is recoverable.
        _save(state_path, pending)
        store.ensure_collection(collection, profile)
        store.upsert(collection, points)
        store.delete(collection, delete_ids)
        _validate_collection(store, collection, expected, profile, records)
        refreshed = load_research_records(root)
        if _records_state(refreshed, {record["research_id"]: chunk_record(record, profile) for record in refreshed}) != expected:
            raise RuntimeError("canonical corpus changed during indexing; retry sync")
        store.switch_alias(profile["alias"], collection)
        pending.update(ready=True, known_point_ids=sorted(target_ids))
        _save(state_path, pending)
        return {"status": "rebuilt" if rebuild else "synced", "records": len(records), "embedded": len(to_embed), "upserted": len(points), "deleted": len(delete_ids), "collection": collection}


def status(*, store, root: Path = RESEARCH_ROOT, state_path: Path = STATE_PATH) -> dict:
    with index_lock(state_path, exclusive=False):
        records = load_research_records(root)
        manifest = read_manifest(state_path)
        if not manifest:
            return {"status": "not_indexed", "canonical_records": len(records)}
        if manifest.get("corpus_root") != str(root.resolve()):
            raise ValueError("manifest belongs to another corpus directory")
        current = {record["research_id"]: (record["source_hash"], record["path"]) for record in records}
        indexed = {key: (value["source_hash"], value["path"]) for key, value in manifest["records"].items()}
        alias = store.alias_target(manifest["profile"]["alias"])
        exists = store.exists(manifest["collection"])
        return {"status": "ready" if exists and manifest["ready"] and current == indexed and alias == manifest["collection"] else "stale_or_dirty",
                "canonical_records": len(records), "indexed_records": len(indexed), "collection": manifest["collection"],
                "alias_target": alias, "points": store.count(manifest["collection"]) if exists else None, "qdrant_version": store.version(),
                "embedding_model": manifest["profile"]["dense"]["model"], "embedding_revision": manifest["profile"]["dense"]["revision"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sync", "status", "rebuild"))
    args = parser.parse_args()
    try:
        store = QdrantStore()
        result = status(store=store) if args.command == "status" else sync(store=store, embedder=OllamaDense(load_profile()), rebuild=args.command == "rebuild")
    except Exception as exc:
        parser.exit(1, f"knowledge index unavailable: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
