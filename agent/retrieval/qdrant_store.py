"""Official SDK REST boundary. Native BM25 inference stays on the Qdrant server."""

from datetime import date
import os

FILTER_FIELDS = {"market", "topic", "status", "tags", "research_id", "date", "include_superseded"}
PAYLOAD_INDEXES = {**{key: "keyword" for key in ("research_id", "market", "topic", "status", "tags")}, "date": "datetime", "is_superseded": "bool"}


def canonical_filter_payload(record: dict, *, is_superseded: bool) -> dict:
    metadata = record["metadata"]
    return {**{key: metadata[key] for key in ("market", "topic", "status", "tags")},
            "research_id": record["research_id"], "date": f"{metadata['date']}T00:00:00Z",
            "is_superseded": is_superseded}


def filter_spec(filters: dict) -> dict | None:
    if set(filters) - FILTER_FIELDS:
        raise ValueError(f"unknown retrieval filters: {sorted(set(filters) - FILTER_FIELDS)}")
    must = []
    include = filters.get("include_superseded", False)
    if not isinstance(include, bool):
        raise ValueError("include_superseded must be boolean")
    if not include:
        must.append({"key": "is_superseded", "match": {"value": False}})
    for key in ("market", "topic", "status", "research_id"):
        value = filters.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} filter must be a nonempty string")
            must.append({"key": key, "match": {"value": value}})
    tags = filters.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not tag for tag in tags):
            raise ValueError("tags filter must be a nonempty array of strings (all required)")
        must.extend({"key": "tags", "match": {"value": tag}} for tag in tags)
    bounds = filters.get("date")
    if bounds is not None:
        if not isinstance(bounds, dict) or not bounds or set(bounds) - {"gte", "lte"}:
            raise ValueError("date filter must contain ISO-date gte and/or lte")
        try:
            dates = {key: date.fromisoformat(value).isoformat() for key, value in bounds.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("date filter bounds must be ISO dates") from exc
        if "gte" in dates and "lte" in dates and dates["gte"] > dates["lte"]:
            raise ValueError("date.gte cannot be after date.lte")
        must.append({"key": "date", "range": {key: f"{value}T00:00:00Z" for key, value in dates.items()}})
    return {"must": must} if must else None


def matches_filter(payload: dict, condition: dict | None) -> bool:
    """Exact metadata-filter parity for the lexical baseline (not a tokenizer)."""
    for clause in (condition or {}).get("must", []):
        value = payload.get(clause["key"])
        if "match" in clause:
            target = clause["match"]["value"]
            if (target not in value if isinstance(value, list) else value != target):
                return False
        else:
            if value is None:
                return False
            bounds = clause["range"]
            if ("gte" in bounds and value < bounds["gte"]) or ("lte" in bounds and value > bounds["lte"]):
                return False
    return True


def query_spec(query: str, vector: list | None, profile: dict, mode: str, filters: dict) -> dict:
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError("retrieval mode must be bm25, dense or hybrid")
    policy = profile["retrieval"]
    condition = filter_spec(filters)
    branches = []
    if mode in {"dense", "hybrid"}:
        if vector is None:
            raise ValueError("dense query vector is required")
        branches.append({"query": vector, "using": "dense", "filter": condition, "limit": policy["branch_top_k"]})
    if mode in {"bm25", "hybrid"}:
        branches.append({"query": {"text": query, "model": "qdrant/bm25"}, "using": "bm25", "filter": condition, "limit": policy["branch_top_k"]})
    common = {"filter": condition, "limit": policy["chunk_top_k"], "with_payload": True, "with_vector": False}
    if mode == "hybrid":
        return {**common, "prefetch": branches, "query": {"rrf": {"k": policy["rrf_k"]}}}
    return {**common, "query": branches[0]["query"], "using": branches[0]["using"]}


class QdrantStore:
    def __init__(self, *, url: str | None = None):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("install requirements.txt (qdrant-client==1.19.0 required)") from exc
        self.models = models
        self.client = QdrantClient(url=url or os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), prefer_grpc=False, timeout=30)

    def version(self) -> str:
        version = self.client.info().version
        parts = version.split(".")
        if int(parts[0]) != 1 or int(parts[1]) < 19:
            raise RuntimeError(f"Qdrant >=1.19 required for native BM25 and parameterized RRF; got {version}")
        return version

    def exists(self, collection: str) -> bool:
        return self.client.collection_exists(collection)

    def ensure_collection(self, collection: str, profile: dict) -> None:
        """Reconcile interrupted creation without altering incompatible collections."""
        self.version()
        m = self.models
        if not self.exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config={"dense": m.VectorParams(size=1024, distance=m.Distance.COSINE)},
                sparse_vectors_config={"bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)},
            )
        info = self.client.get_collection(collection)
        vectors, sparse = info.config.params.vectors, info.config.params.sparse_vectors
        if (not isinstance(vectors, dict) or set(vectors) != {"dense"} or vectors["dense"].size != profile["dense"]["size"]
                or vectors["dense"].distance != m.Distance.COSINE or not isinstance(sparse, dict)
                or set(sparse) != {"bm25"} or sparse["bm25"].modifier != m.Modifier.IDF):
            raise RuntimeError("index collection vector configuration mismatch; rebuild required")
        for key, schema in PAYLOAD_INDEXES.items():
            existing = info.payload_schema.get(key)
            if existing is not None and existing.data_type != m.PayloadSchemaType(schema):
                raise RuntimeError(f"index payload schema mismatch: {key}; rebuild required")
            if existing is None:
                self.client.create_payload_index(collection_name=collection, field_name=key, field_schema=m.PayloadSchemaType(schema), wait=True)
        verified = self.client.get_collection(collection)
        if any(key not in verified.payload_schema or verified.payload_schema[key].data_type != m.PayloadSchemaType(schema) for key, schema in PAYLOAD_INDEXES.items()):
            raise RuntimeError("index payload indexes incomplete; retry sync")

    def upsert(self, collection: str, points: list[dict]) -> None:
        if points:
            # Generated SDK REST API deliberately bypasses client-side Document/FastEmbed inference.
            self.client.http.points_api.upsert_points(
                collection_name=collection, wait=True,
                point_insert_operations=self.models.PointsList.model_validate({"points": points}),
            )

    def delete(self, collection: str, point_ids: list[str]) -> None:
        if point_ids:
            self.client.delete(collection_name=collection, points_selector=self.models.PointIdsList(points=point_ids), wait=True)

    def count(self, collection: str) -> int:
        return self.client.count(collection_name=collection, exact=True).count

    def points(self, collection: str, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        return [point.model_dump(mode="json") for point in self.client.retrieve(collection_name=collection, ids=ids, with_payload=True, with_vectors=True)]

    def query(self, collection: str, request: dict) -> list[dict]:
        response = self.client.http.search_api.query_points(
            collection_name=collection,
            query_request=self.models.QueryRequest.model_validate(request),
        )
        return [point.model_dump(mode="json") for point in response.result.points]

    def alias_target(self, alias: str) -> str | None:
        return next((item.collection_name for item in self.client.get_aliases().aliases if item.alias_name == alias), None)

    def switch_alias(self, alias: str, collection: str) -> None:
        m = self.models
        operations = []
        if self.alias_target(alias):
            operations.append(m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=alias)))
        operations.append(m.CreateAliasOperation(create_alias=m.CreateAlias(alias_name=alias, collection_name=collection)))
        self.client.update_collection_aliases(change_aliases_operations=operations)
