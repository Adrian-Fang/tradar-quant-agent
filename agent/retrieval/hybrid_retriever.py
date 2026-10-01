"""Explicit filtered search, record collapse, and canonical-source hydration."""

from pathlib import Path
from time import perf_counter

from .chunking import chunk_record, load_profile, profile_hash
from .embeddings import OllamaDense, validate_vectors
from .ingest import STATE_PATH, index_lock, read_manifest, resolved_profile
from .loader import RESEARCH_ROOT, load_research_records
from .qdrant_store import QdrantStore, canonical_filter_payload, filter_spec, matches_filter, query_spec


class KnowledgeRetriever:
    def __init__(self, *, store=None, embedder=None, profile: dict | None = None,
                 root: Path = RESEARCH_ROOT, state_path: Path = STATE_PATH):
        self.profile = profile or load_profile()
        self.store = store if store is not None else QdrantStore()
        self.embedder = embedder if embedder is not None else OllamaDense(self.profile)
        self.root, self.state_path = root, state_path

    def retrieve(self, query: str, *, mode: str = "hybrid", limit: int = 5, **filters) -> list[dict]:
        return self.search(query, mode=mode, limit=limit, **filters)["results"]

    def search(self, query: str, *, mode: str = "hybrid", limit: int = 5, **filters) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if type(limit) is not int or not 1 <= limit <= self.profile["retrieval"]["record_top_k"]:
            raise ValueError("record limit must be between 1 and 5")
        condition = filter_spec(filters)  # Reject unknown filters before any service calls.
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("retrieval mode must be bm25, dense or hybrid")
        started, latency = perf_counter(), {}
        with index_lock(self.state_path, exclusive=False):
            manifest = read_manifest(self.state_path)
            if not manifest or not manifest.get("ready"):
                raise RuntimeError("knowledge index unavailable/dirty; run ingest sync or rebuild")
            if manifest.get("corpus_root") != str(self.root.resolve()):
                raise RuntimeError("index corpus directory mismatch")
            profile = resolved_profile(self.profile, manifest["profile"]["dense"]["revision"])
            if manifest["profile_hash"] != profile_hash(profile):
                raise RuntimeError("index profile mismatch; rebuild required")
            records = load_research_records(self.root)
            canonical = {record["research_id"]: record for record in records}
            current = {key: (record["source_hash"], record["path"]) for key, record in canonical.items()}
            indexed = {key: (record["source_hash"], record["path"]) for key, record in manifest["records"].items()}
            if current != indexed:
                raise RuntimeError("stale knowledge index: canonical source hash/path mismatch; run ingest sync")
            if self.store.alias_target(profile["alias"]) != manifest["collection"]:
                raise RuntimeError("knowledge index alias changed; reconcile manifest")
            latency["canonical_validation_ms"] = (perf_counter() - started) * 1000
            vector = None
            tick = perf_counter()
            if mode != "bm25":
                if self.embedder.revision != profile["dense"]["revision"]:
                    raise RuntimeError("embedding model revision changed; rebuild required")
                text = f"Instruct: {profile['dense']['query_instruction']}\nQuery: {query}"
                vector = validate_vectors(self.embedder([text]), 1)[0]
                if self.embedder.revision != profile["dense"]["revision"]:
                    raise RuntimeError("embedding model revision changed during query embedding")
            latency["embedding_ms"] = (perf_counter() - tick) * 1000
            tick = perf_counter()
            points = self.store.query(manifest["collection"], query_spec(query, vector, profile, mode, filters))
            latency["qdrant_ms"] = (perf_counter() - tick) * 1000
            tick = perf_counter()
            # ponytail: authoritative scans suit this small corpus; add a file cache only after measuring.
            refreshed = load_research_records(self.root)
            if {record["research_id"]: (record["source_hash"], record["path"]) for record in refreshed} != current:
                raise RuntimeError("canonical corpus changed during retrieval; retry after sync")
            expected_chunks = {chunk["point_id"]: (record, chunk) for record in records for chunk in chunk_record(record, profile)}
            superseded = {key for record in records for key in record["metadata"]["supersedes"]}
            collapsed = {}
            for point in points:
                pair = expected_chunks.get(str(point["id"]))
                payload = point.get("payload", {})
                if not pair:
                    raise RuntimeError("stale knowledge index: unknown/deleted point")
                record, chunk = pair
                metadata = canonical_filter_payload(record, is_superseded=record["metadata"]["status"] == "superseded" or record["research_id"] in superseded)
                checks = {"research_id": record["research_id"], "source_path": record["path"], "source_hash": record["source_hash"],
                          "profile_hash": profile_hash(profile), "embedding_revision": profile["dense"]["revision"],
                          **{key: chunk[key] for key in ("chunk_id", "chunk_hash", "embedding_input_hash", "text")}, **metadata}
                if any(type(payload.get(key)) is not type(value) or payload[key] != value for key, value in checks.items()):
                    raise RuntimeError(f"stale knowledge index: payload/source mismatch: {chunk['chunk_id']}")
                if not matches_filter(metadata, condition):
                    raise RuntimeError(f"knowledge index violated canonical filters: {chunk['chunk_id']}")
                research_id = record["research_id"]
                if research_id not in collapsed:
                    collapsed[research_id] = {**record, "score": point["score"], "matched_chunks": []}
                collapsed[research_id]["matched_chunks"].append({"chunk_id": chunk["chunk_id"], "score": point["score"]})
            # First fused occurrence wins. Do not re-sort records or treat similarity as support.
            results = list(collapsed.values())[:limit]
            latency["hydration_ms"] = (perf_counter() - tick) * 1000
        latency["total_ms"] = (perf_counter() - started) * 1000
        return {"results": results, "latency_ms": latency, "mode": mode, "chunks_returned": len(points)}
