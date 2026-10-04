"""Hermetic AE-14 contracts; no Qdrant/Ollama/provider service required."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.retrieval.chunking import chunk_record, load_profile, profile_hash
from agent.retrieval.embeddings import validate_vectors
from agent.retrieval.embeddings import OllamaDense
from agent.retrieval.eval import CASES, compare_retrievers, run_case
from agent.retrieval.hybrid_retriever import KnowledgeRetriever
from agent.retrieval.ingest import read_manifest, status, sync
from agent.retrieval.loader import RESEARCH_ROOT, load_research_records
from agent.retrieval.qdrant_store import PAYLOAD_INDEXES, QdrantStore, filter_spec, matches_filter, query_spec
from agent.retrieval.relevance_verifier import verify_candidates


class FakeEmbedder:
    revision = "sha256:fixture-model"

    def __init__(self):
        self.inputs = []

    def __call__(self, texts):
        self.inputs.extend(texts)
        return [[1.0] + [0.0] * 1023 for _ in texts]


class FakeStore:
    """Storage/REST boundary double, not a substitute for live BM25 quality tests."""

    def __init__(self):
        self.collections, self.aliases, self.calls = {}, {}, []
        self.fail_upsert = False
        self.fail_query = False
        self.order = None

    def version(self):
        return "1.19.1-fixture"

    def exists(self, collection):
        return collection in self.collections

    def create(self, collection, profile):
        assert collection not in self.collections
        self.collections[collection] = {}
        self.calls.append(("create", collection))

    def ensure_collection(self, collection, profile):
        if not self.exists(collection):
            self.create(collection, profile)

    def upsert(self, collection, points):
        self.calls.append(("upsert", deepcopy(points)))
        for point in points:
            value = deepcopy(point)
            assert value["vector"]["bm25"]["model"] == "qdrant/bm25"
            value["vector"]["bm25"] = {"indices": [1], "values": [1.0]}
            self.collections[collection][point["id"]] = value
            if self.fail_upsert:
                raise RuntimeError("partial remote write")

    def delete(self, collection, ids):
        self.calls.append(("delete", list(ids)))
        for point_id in ids:
            self.collections[collection].pop(point_id, None)

    def count(self, collection):
        return len(self.collections[collection])

    def points(self, collection, ids):
        return [deepcopy(self.collections[collection][key]) for key in ids if key in self.collections[collection]]

    def query(self, collection, request):
        self.calls.append(("query", deepcopy(request)))
        if self.fail_query:
            raise RuntimeError("native BM25/RRF unsupported")
        if "prefetch" in request:
            assert request["query"] == {"rrf": {"k": 60}}
            assert len(request["prefetch"]) == 2
            assert all(branch["filter"] == request["filter"] for branch in request["prefetch"])
            assert all(branch["limit"] == 20 for branch in request["prefetch"])
        values = list(self.collections[collection].values())
        if self.order is not None:
            values = [self.collections[collection][key] for key in self.order]
        values = [point for point in values if matches_filter(point["payload"], request["filter"])]
        return [{**deepcopy(point), "score": 1.0 / (index + 1)} for index, point in enumerate(values[:request["limit"]])]

    def alias_target(self, alias):
        return self.aliases.get(alias)

    def switch_alias(self, alias, collection):
        self.calls.append(("alias", collection))
        self.aliases[alias] = collection


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    for path in (RESEARCH_ROOT / "rr-001-relative-reversal.md", RESEARCH_ROOT / "rr-010-trading-cost-defaults.md"):
        (root / path.name).write_bytes(path.read_bytes())
    return {"root": root, "state_path": tmp_path / "state/manifest.json", "store": FakeStore(), "embedder": FakeEmbedder(), "profile": load_profile()}


def _edit(backend, old, new, name="rr-001-relative-reversal.md"):
    path = backend["root"] / name
    path.write_text(path.read_text().replace(old, new), encoding="utf-8")


def test_canonical_legacy_records_and_three_semantic_chunks():
    records = load_research_records()
    assert len(records) == 10
    chunks = [chunk for record in records for chunk in chunk_record(record, load_profile())]
    assert len(chunks) == len({chunk["point_id"] for chunk in chunks}) == 30
    for record in records:
        assert record["legacy_adapter"] and record["schema_version"] == "1.0"
        assert record["metadata"]["source_task_id"] is None
        assert isinstance(record["source_ref"], list)
        assert len(record["source_hash"]) == 64
        assert all("# Provenance" not in chunk["text"] for chunk in chunk_record(record, load_profile()))


@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_hydrated_qdrant_records_reach_agent_context_without_live_services(backend, strategy):
    from agent.agent import run_agent

    sync(**backend)
    retriever = KnowledgeRetriever(**backend)
    verifier = Mock(provider="fixture", model="fixture")
    verifier.create.return_value = {"output_text": json.dumps({"supported": True, "reason": "cost defaults directly support the query"})}
    planner = Mock(provider="fixture", model="fixture")
    planner.create.side_effect = [
        {"output": [{"type": "function_call", "call_id": "lookup", "name": "search_knowledge",
                     "arguments": json.dumps({"query": "What are the canonical transaction cost assumptions?", "answer_target": None})}]},
        {"output_text": "Use recorded costs."},
    ]
    synthesis = Mock(provider="fixture", model="fixture")
    synthesis.create.return_value = {"output_text": json.dumps({"status": "success", "answer": "10 bp buy and 15 bp sell [knowledge-RR-010]", "evidence_ids": ["knowledge-RR-010"]})}
    grounding = Mock(provider="fixture", model="fixture")
    grounding.create.return_value = {"output_text": json.dumps({"answer": "ignored", "claims": [{"claim": "10 bp buy and 15 bp sell", "evidence_ids": ["knowledge-RR-010"], "grounding": "supported"}]})}
    result = run_agent(
        "What are the canonical transaction cost assumptions?", planner_client=planner,
        retrieval_backend="qdrant", knowledge_retriever=retriever, retrieval_client=verifier,
        synthesis_client=synthesis, grounding_client=grounding,
        retrieval_strategy=strategy,
        retrieval_filters={"research_id": "RR-010"}, candidate_limit=1,
    )
    record = next(record for record in load_research_records(backend["root"]) if record["research_id"] == "RR-010")
    assert result["retrieval"]["results"][0]["source_hash"] == record["source_hash"]
    assert result["observed"]["retrieval"]["research_ids"] == ["RR-010"]
    assert result["observed"]["retrieval"]["chunks_returned"] == 3
    assert result["observed"]["retrieval"]["mode"] == strategy
    request = next(call[1] for call in reversed(backend["store"].calls) if call[0] == "query")
    if strategy == "dense":
        assert request["using"] == "dense" and "prefetch" not in request
    else:
        assert request["query"] == {"rrf": {"k": 60}} and len(request["prefetch"]) == 2
    verified = json.loads(verifier.create.call_args.args[0]["input"])["research_record"]
    assert verified["text"] == record["text"] and "score" not in verified
    observation = json.loads(planner.create.call_args.args[0]["input"][-1]["output"])
    brief = observation["records"][0]
    assert brief["context_type"] == "related_research_context"
    assert brief["question"] == record["question"][:240]
    assert brief["conclusion"] == record["sections"]["Conclusion"][:200]
    assert "provenance" not in brief and "# Provenance" not in str(brief)
    assert result["status"] == "ok"
    assert result["observed"]["outcome"]["status"] == "success"
    assert result["research_run"] is None and result["hitl"] is None
    assert result["grounding"]["fully_grounded"]
    knowledge = json.loads(result["evidence"][0]["text"])
    assert knowledge["evidence_type"] == "knowledge_record"
    assert knowledge["provenance"]["source_hash"] == record["source_hash"]
    assert "score" not in knowledge and knowledge["content"] == record["text"]


def test_public_corpus_has_no_private_messaging_identifiers():
    import re
    private = re.compile(r"(?i:\bslack\b)|\b[CDGU][A-Z0-9]{10}\b|\b\d{10}\.\d{6}\b")
    records = load_research_records()
    for path in RESEARCH_ROOT.glob("*.md"):
        assert not private.search(path.read_text()), path.name
    for record in records:
        assert record["source"] == "research_record"
        assert record["metadata"]["source_task_id"] is None
        for reference in record["source_ref"]:
            assert reference.startswith("arxiv:") or (RESEARCH_ROOT.parents[2] / reference).is_file()


@pytest.mark.parametrize("task_field", ["", "source_task_id: null\n"])
@pytest.mark.parametrize("ref_field", ["", "source_ref: []\n"])
def test_versioned_record_can_disclose_unknown_provenance_without_private_ids(backend, task_field, ref_field):
    _edit(backend, "research_id: RR-001", f'schema_version: "1.0"\n{task_field}research_id: RR-001')
    _edit(backend, "source_ref: []\n", ref_field)
    record = load_research_records(backend["root"])[0]
    assert record["metadata"]["source_task_id"] is None
    assert record["source_ref"] == [] and record["provenance"]
    assert not record["legacy_adapter"]
    collection = sync(**backend)["collection"]
    points = backend["store"].collections[collection]
    assert all(point["payload"]["source_task_id"] is None for point in points.values())


def test_yaml_quotes_scalar_refs_and_versioned_contract(backend):
    _edit(backend, "tags: [relative-strength, short-term-reversal, cross-sectional, factor]", "tags: ['relative,strength', '中文标签']")
    _edit(backend, "research_id: RR-001", 'schema_version: "1.0"\nsource_task_id: published-study-1\nresearch_id: RR-001')
    _edit(backend, "source_ref: []", "source_ref: data/factor_defs/high52.yaml")
    record = load_research_records(backend["root"])[0]
    assert record["tags"] == ["relative,strength", "中文标签"]
    assert not record["legacy_adapter"]
    assert record["source_ref"] == ["data/factor_defs/high52.yaml"]
    assert isinstance(load_research_records(backend["root"])[1]["source_ref"], list)


@pytest.mark.parametrize("old,new,error", [
    ("research_id: RR-001", "research_id: 1", "research_id"),
    ("market: A-share", "market: []", "market"),
    ("status: inconclusive", "status: awesome", "status"),
    ("date: 2026-08-31", "date: yesterday", "date"),
    ("research_id: RR-001", 'schema_version: "2.0"\nresearch_id: RR-001', "schema_version"),
    ("research_id: RR-001", 'schema_version: "1.0"\nsource_task_id: 1\nresearch_id: RR-001', "source_task_id"),
    ("supersedes: []", "supersedes: RR-000", "supersedes"),
    ("source_type: research_record", "source_type: research_record\nsource_type: other", "duplicate"),
    ("source_ref: []", "source_ref: null", "source_ref"),
    ("source_ref: []", "source_ref: {path: example}", "source_ref"),
    ("# Method", "# Missing Method", "section"),
    ("# Method", "# Research Question", "duplicate"),
    ("# Caveats", "# Conclusion", "duplicate"),
    ("# Provenance", "# Provenance\n\n```", "fenced"),
])
def test_record_contract_rejects_malformed(backend, old, new, error):
    _edit(backend, old, new)
    with pytest.raises(ValueError, match=error):
        load_research_records(backend["root"])


def test_duplicate_ids_and_missing_corpus(backend):
    _edit(backend, "research_id: RR-010", "research_id: RR-001", "rr-010-trading-cost-defaults.md")
    with pytest.raises(ValueError, match="duplicate research_id"):
        load_research_records(backend["root"])
    with pytest.raises(ValueError, match="directory unavailable"):
        load_research_records(backend["root"] / "missing")


def test_chunk_identity_tables_and_localized_hash_changes(backend):
    record = load_research_records(backend["root"])[0]
    old = chunk_record(record, backend["profile"])
    table = "| horizon | mean |\n| --- | --- |\n| 5 | 0.01 |"
    record["sections"]["Key Findings"] += "\n\n" + table
    new = chunk_record(record, backend["profile"])
    assert old[0] == new[0] and old[2] == new[2]
    assert old[1]["point_id"] == new[1]["point_id"]
    assert old[1]["chunk_hash"] != new[1]["chunk_hash"]
    assert table in new[1]["text"]
    record["sections"]["Key Findings"] = "x" * 25000
    with pytest.raises(ValueError, match="subdivision"):
        chunk_record(record, backend["profile"])


@pytest.mark.parametrize("vector", [[1.0], [0.0] * 1024, [float("nan")] * 1024, [True] * 1024])
def test_bad_embeddings_rejected(vector):
    with pytest.raises(ValueError, match="embedding"):
        validate_vectors([vector], 1)


@pytest.mark.parametrize("filters", [
    {"language": "zh"}, {"market": []}, {"research_id": ""}, {"tags": "cost"},
    {"tags": []}, {"include_superseded": 1}, {"date": "2026-01-01"},
    {"date": {"gt": "2026-01-01"}}, {"date": {"gte": 12}},
    {"date": {"gte": "2026-02-01", "lte": "2026-01-01"}},
])
def test_invalid_explicit_filters_rejected(filters):
    with pytest.raises(ValueError):
        filter_spec(filters)


def test_native_query_request_shape():
    filters = {"market": "A-share", "status": "validated", "tags": ["buy-cost"], "date": {"gte": "2026-01-01"}}
    query = query_spec("成本", [1.0] * 1024, load_profile(), "hybrid", filters)
    assert query["query"] == {"rrf": {"k": 60}}
    assert query["limit"] == 20
    assert query["prefetch"][1]["query"] == {"text": "成本", "model": "qdrant/bm25"}
    assert all(branch["filter"] == filter_spec(filters) for branch in query["prefetch"])
    assert query_spec("成本", None, load_profile(), "bm25", {})["using"] == "bm25"


def test_sdk_models_accept_native_bm25_and_rrf():
    models = pytest.importorskip("qdrant_client.models", reason="SDK installation blocked by host DNS; no fake SDK validation")
    request = query_spec("research", [1.0] * 1024, load_profile(), "hybrid", {"date": {"gte": "2026-01-01"}})
    validated = models.QueryRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    assert validated["query"] == {"rrf": {"k": 60}}
    points = models.PointsList.model_validate({"points": [{"id": "cb159c08-7563-5737-a3d4-7343c4c8ef40", "vector": {"dense": [1.0] * 1024, "bm25": {"text": "research", "model": "qdrant/bm25"}}}]})
    assert points.points[0].vector["bm25"].model == "qdrant/bm25"
    # Exercise the real SDK's generated endpoint without any network or inference.
    import httpx
    from qdrant_client import QdrantClient
    calls = []

    def respond(request):
        calls.append((request.method, request.url.path, json.loads(request.content)))
        result = {"points": [{"id": points.points[0].id, "version": 0, "score": 1.0, "payload": {}}]} if request.url.path.endswith("/query") else {"operation_id": 1, "status": "completed"}
        return httpx.Response(200, json={"result": result, "status": "ok", "time": 0.001})

    store = QdrantStore.__new__(QdrantStore)
    store.models = models
    store.client = QdrantClient(url="http://fixture", check_compatibility=False, transport=httpx.MockTransport(respond))
    try:
        store.upsert("fixture", [points.points[0].model_dump(mode="json", exclude_none=True)])
        for mode in ("dense", "bm25", "hybrid"):
            assert store.query("fixture", query_spec("research", [1.0] * 1024, load_profile(), mode, {}))[0]["score"] == 1
    finally:
        store.client.close()
    assert calls[0][0:2] == ("PUT", "/collections/fixture/points")
    assert calls[0][2]["points"][0]["vector"]["bm25"]["model"] == "qdrant/bm25"
    assert all(call[0:2] == ("POST", "/collections/fixture/points/query") for call in calls[1:])
    assert calls[-1][2]["query"] == {"rrf": {"k": 60}}


def test_query_adapter_uses_search_api_not_points_api():
    store = QdrantStore.__new__(QdrantStore)
    point = SimpleNamespace(model_dump=Mock(return_value={"id": "fixture", "score": 1.0}))
    endpoint = Mock(return_value=SimpleNamespace(result=SimpleNamespace(points=[point])))
    store.client = SimpleNamespace(http=SimpleNamespace(search_api=SimpleNamespace(query_points=endpoint), points_api=SimpleNamespace()))
    store.models = SimpleNamespace(QueryRequest=SimpleNamespace(model_validate=Mock(side_effect=lambda value: value)))
    for mode in ("dense", "bm25", "hybrid"):
        request = query_spec("research", [1.0] * 1024, load_profile(), mode, {})
        assert store.query("fixture", request) == [{"id": "fixture", "score": 1.0}]
        endpoint.assert_called_with(collection_name="fixture", query_request=request)


def test_collection_and_payload_indexes_recover_interrupted_creation():
    store = QdrantStore.__new__(QdrantStore)
    store.models = SimpleNamespace(Distance=SimpleNamespace(COSINE="Cosine"), Modifier=SimpleNamespace(IDF="idf"),
                                   VectorParams=lambda **kwargs: SimpleNamespace(**kwargs), SparseVectorParams=lambda **kwargs: SimpleNamespace(**kwargs), PayloadSchemaType=lambda value: value)
    indexes, created = {}, []
    params = SimpleNamespace(vectors={"dense": SimpleNamespace(size=1024, distance="Cosine")}, sparse_vectors={"bm25": SimpleNamespace(modifier="idf")})
    info = SimpleNamespace(config=SimpleNamespace(params=params), payload_schema=indexes)

    def create_index(**kwargs):
        field = kwargs["field_name"]
        if field == "date" and "date" not in created:
            created.append("date")
            raise RuntimeError("interrupted payload index creation")
        indexes[field] = SimpleNamespace(data_type=kwargs["field_schema"])

    store.client = SimpleNamespace(info=lambda: SimpleNamespace(version="1.19.1"),
                                   collection_exists=lambda name: bool(created), create_collection=lambda **kwargs: created.append("collection"),
                                   get_collection=lambda name: info, create_payload_index=Mock(side_effect=create_index))
    with pytest.raises(RuntimeError, match="interrupted"):
        store.ensure_collection("fixture", load_profile())
    assert len(indexes) == 5
    store.client.create_payload_index.reset_mock()
    store.ensure_collection("fixture", load_profile())
    assert created.count("collection") == 1 and set(indexes) == set(PAYLOAD_INDEXES)
    assert store.client.create_payload_index.call_count == 2
    store.client.create_payload_index.reset_mock()
    store.ensure_collection("fixture", load_profile())
    store.client.create_payload_index.assert_not_called()
    indexes["research_id"].data_type = "integer"
    with pytest.raises(RuntimeError, match="payload schema mismatch"):
        store.ensure_collection("fixture", load_profile())
    params.vectors["dense"].size = 384
    with pytest.raises(RuntimeError, match="vector configuration mismatch"):
        store.ensure_collection("fixture", load_profile())


def test_incremental_sync_noops_and_changed_chunk_only_embedding(backend):
    first = sync(**backend)
    assert first["embedded"] == first["upserted"] == 6
    manifest = read_manifest(backend["state_path"])
    assert manifest["ready"] and manifest["profile_hash"] == profile_hash(manifest["profile"])
    backend["embedder"].inputs.clear()
    backend["store"].calls.clear()
    assert sync(**backend)["status"] == "unchanged"
    assert backend["embedder"].inputs == backend["store"].calls == []
    _edit(backend, "# Key Findings", "# Key Findings\n\nAdditional deterministic test finding.")
    changed = sync(**backend)
    assert changed["embedded"] == 1 and changed["upserted"] == 3
    assert len(backend["embedder"].inputs) == 1
    assert status(**{key: backend[key] for key in ("store", "root", "state_path")})["status"] == "ready"


def test_metadata_provenance_change_updates_without_embedding(backend):
    sync(**backend)
    backend["embedder"].inputs.clear()
    _edit(backend, "# Provenance", "# Provenance\n\nAdditional public provenance note.")
    result = sync(**backend)
    assert result["embedded"] == 0 and result["upserted"] == 3
    assert not backend["embedder"].inputs


def test_deletion_only_after_successful_scan_and_known_ids_only(backend):
    first = sync(**backend)
    collection = first["collection"]
    ids = set(backend["store"].collections[collection])
    (backend["root"] / "rr-010-trading-cost-defaults.md").unlink()
    _edit(backend, "# Method", "# Missing Method")
    calls = len(backend["store"].calls)
    with pytest.raises(ValueError, match="section"):
        sync(**backend)
    assert len(backend["store"].calls) == calls
    assert set(backend["store"].collections[collection]) == ids
    _edit(backend, "# Missing Method", "# Method")
    result = sync(**backend)
    assert result["deleted"] == 3 and result["embedded"] == 0
    assert backend["store"].count(collection) == 3


def test_partial_write_journal_recovers_orphans(backend):
    backend["store"].fail_upsert = True
    with pytest.raises(RuntimeError, match="partial remote"):
        sync(**backend)
    manifest = read_manifest(backend["state_path"])
    assert not manifest["ready"] and len(manifest["known_point_ids"]) == 6
    (backend["root"] / "rr-001-relative-reversal.md").unlink()
    backend["store"].fail_upsert = False
    result = sync(**backend)
    assert result["deleted"] == 3 and result["upserted"] == 3
    assert read_manifest(backend["state_path"])["ready"]


def test_rebuild_switches_only_after_validation_and_keeps_old_collection(backend):
    old = sync(**backend)["collection"]
    backend["store"].fail_query = True
    with pytest.raises(RuntimeError, match="unsupported"):
        sync(**backend, rebuild=True)
    assert backend["store"].aliases["tradar_research"] == old
    assert not read_manifest(backend["state_path"])["ready"]
    backend["store"].fail_query = False
    new = sync(**backend, rebuild=True)["collection"]
    assert new != old and old in backend["store"].collections
    assert backend["store"].aliases["tradar_research"] == new


def test_profile_or_model_revision_change_requires_rebuild(backend):
    sync(**backend)
    backend["embedder"].revision = "new-model-digest"
    with pytest.raises(ValueError, match="rebuild"):
        sync(**backend)
    assert sync(**backend, rebuild=True)["status"] == "rebuilt"


def test_full_fake_ingestion_hybrid_hydration_verification_and_eval(backend):
    first = sync(**backend)
    retriever = KnowledgeRetriever(**backend)
    points = backend["store"].collections[first["collection"]]
    # RRF fused fixture order: cost record, all three reversal chunks, other cost chunks.
    ids = list(points)
    backend["store"].order = [ids[3], ids[0], ids[1], ids[2], ids[4], ids[5]]
    candidates = retriever.search("成本与反转")
    assert [record["research_id"] for record in candidates["results"]] == ["RR-010", "RR-001"]
    assert len(candidates["results"][0]["matched_chunks"]) == 3
    assert "# Provenance" in candidates["results"][0]["text"]
    assert set(candidates["latency_ms"]) == {"canonical_validation_ms", "embedding_ms", "qdrant_ms", "hydration_ms", "total_ms"}
    assert backend["embedder"].inputs[-1].startswith("Instruct:")

    class Client:
        def create(self, payload):
            body = json.loads(payload["input"])
            records = body.get("research_records", [body.get("research_record")])
            decisions = []
            for record in records:
                assert "score" not in record and "matched_chunks" not in record
                assert "# Method" in record["text"] and "# Provenance" in record["text"]
                decisions.append({"research_id": record["research_id"], "supported": record["research_id"] == "RR-001", "reason": "fixture support"})
            value = {"results": decisions} if "research_records" in body else {key: decisions[0][key] for key in ("supported", "reason")}
            return {"output_text": json.dumps(value)}

    def verified(query, **filters):
        return verify_candidates(query, retriever.retrieve(query, **filters), client=Client())

    cases = [{"id": "positive", "query": "反转", "relevant_ids": ["RR-001"]},
             {"id": "deleted", "query": "deleted", "research_id": "RR-DELETED", "relevant_ids": []}]
    from agent.retrieval.lexical_retriever import retrieve
    from unittest.mock import patch
    with patch("agent.retrieval.lexical_retriever.load_research_records", return_value=load_research_records(backend["root"])):
        evaluations = compare_retrievers({
            "lexical": retrieve,
            "bm25": lambda query, **filters: retriever.search(query, mode="bm25", **filters),
            "dense": lambda query, **filters: retriever.search(query, mode="dense", **filters),
            "hybrid": retriever.search, "hybrid+verification": verified,
        }, cases=cases, measure_latency=False)
    assert len(evaluations) == 5
    assert evaluations["hybrid+verification"]["macro"]["mrr@5"] == 1
    assert evaluations["hybrid"]["macro"]["mrr@5"] == 0.5
    assert all(item["macro"]["no_relevance"]["abstention_rate"] == 1 for item in evaluations.values())


def test_explicit_filters_and_superseding_cascade(backend):
    sync(**backend)
    _edit(backend, "supersedes: []", "supersedes: [RR-001]", "rr-010-trading-cost-defaults.md")
    result = sync(**backend)
    assert result["upserted"] == 6 and result["embedded"] == 0
    retriever = KnowledgeRetriever(**backend)
    assert [record["research_id"] for record in retriever.retrieve("research", mode="bm25")] == ["RR-010"]
    assert len(retriever.retrieve("research", mode="bm25", include_superseded=True)) == 2
    assert retriever.retrieve("research", research_id="RR-001") == []
    assert retriever.retrieve("research", market="US-equity") == []
    assert [record["research_id"] for record in retriever.retrieve("research", tags=["buy-cost"], status="validated", date={"gte": "2026-01-01"})] == ["RR-010"]
    _edit(backend, "supersedes: [RR-001]", "supersedes: []", "rr-010-trading-cost-defaults.md")
    sync(**backend)
    assert len(retriever.retrieve("research")) == 2


@pytest.mark.parametrize("change", ["edit", "delete", "rename", "dirty", "alias", "payload", "revision"])
def test_retrieval_fails_closed_on_stale_or_tampered_index(backend, change):
    first = sync(**backend)
    retriever = KnowledgeRetriever(**backend)
    path = backend["root"] / "rr-001-relative-reversal.md"
    if change == "edit":
        _edit(backend, "# Conclusion", "# Conclusion\n\nChanged conclusion.")
    elif change == "delete":
        path.unlink()
    elif change == "rename":
        path.rename(path.with_name("new-name.md"))
    elif change == "dirty":
        manifest = read_manifest(backend["state_path"])
        manifest["ready"] = False
        backend["state_path"].write_text(json.dumps(manifest))
    elif change == "alias":
        backend["store"].aliases["tradar_research"] = "other-collection"
    elif change == "payload":
        point = next(iter(backend["store"].collections[first["collection"]].values()))
        point["payload"]["text"] = "tampered chunk"
    else:
        backend["embedder"].revision = "new-model"
    with pytest.raises(RuntimeError):
        retriever.retrieve("research")


def test_unique_record_metrics_and_mrr_at_five():
    case = {"id": "dedup", "relevant_ids": ["RR-B"]}
    row = run_case(case, ["RR-A", "RR-A", "RR-B"], (1, 3, 5), [1, .9, .8])
    assert row["top_results"] == ["RR-A", "RR-B"]
    assert row["reciprocal_rank@5"] == .5
    assert row["precision@3"] == 1 / 3
    row = run_case(case, [f"RR-{index}" for index in range(5)] + ["RR-B"], (1, 3, 5))
    assert row["reciprocal_rank@5"] == 0
    assert len(CASES) == 23  # Preserve previous evaluation oracle.


def test_verifier_failure_not_counted_as_successful_abstention():
    evaluations = compare_retrievers({"broken": lambda query, **filters: {"status": "error", "results": [], "errors": ["provider"]}},
                                    cases=[{"id": "none", "query": "none", "relevant_ids": []}], measure_latency=False)
    assert evaluations["broken"]["macro"]["error_count"] == 1
    negative = evaluations["broken"]["macro"]["no_relevance"]
    assert negative["abstention_rate"] is negative["false_positive_rate"] is None
    assert negative["valid_count"] == negative["empty_result_count"] == 0
    assert negative["error_count"] == 1


@pytest.mark.parametrize("corruption", ["json", "namespace", "journal", "profile"])
def test_corrupted_manifest_blocks_sync_but_rebuild_recovers(backend, corruption):
    old = sync(**backend)["collection"]
    manifest = read_manifest(backend["state_path"])
    if corruption == "json":
        backend["state_path"].write_text("{")
    else:
        if corruption == "namespace":
            manifest["collection"] = "unrelated_collection"
        elif corruption == "journal":
            manifest["known_point_ids"] = []
        else:
            manifest["profile_hash"] = "bad"
        backend["state_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        sync(**backend)
    assert backend["store"].aliases["tradar_research"] == old
    assert sync(**backend, rebuild=True)["status"] == "rebuilt"


@pytest.mark.parametrize("operation", ["sync", "search"])
def test_concurrent_source_edit_fails_closed(backend, operation):
    if operation == "search":
        sync(**backend)
    original = backend["store"].query

    def query_with_edit(collection, request):
        points = original(collection, request)
        _edit(backend, "# Conclusion", "# Conclusion\n\nConcurrent edit.")
        return points

    backend["store"].query = query_with_edit
    with pytest.raises(RuntimeError, match="changed during"):
        sync(**backend) if operation == "sync" else KnowledgeRetriever(**backend).retrieve("research")
    if operation == "sync":
        assert not read_manifest(backend["state_path"])["ready"]
        assert backend["store"].aliases == {}


def test_embedding_revision_change_during_batch_is_not_published(backend):
    class ChangingEmbedder(FakeEmbedder):
        def __call__(self, texts):
            result = super().__call__(texts)
            self.revision = "changed-during-embedding"
            return result

    backend["embedder"] = ChangingEmbedder()
    with pytest.raises(RuntimeError, match="revision changed during"):
        sync(**backend)
    assert backend["store"].calls == []
    assert not backend["state_path"].exists()


def test_ollama_adapter_pins_digest_disables_truncation_and_never_pulls(monkeypatch):
    from io import StringIO
    calls = []

    def fake_open(url, timeout):
        calls.append(url)
        return StringIO(json.dumps({"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:model"}]}))

    monkeypatch.setattr("agent.retrieval.embeddings.request.urlopen", fake_open)
    embedder = OllamaDense(load_profile(), base_url="http://fixture")
    assert embedder.revision == "sha256:model"
    assert embedder.client.truncate is False
    assert calls == ["http://fixture/api/tags"]
    monkeypatch.setattr("agent.retrieval.embeddings.request.urlopen", lambda url, timeout: StringIO('{"models": []}'))
    with pytest.raises(RuntimeError, match="manually run: ollama pull qwen3-embedding:0.6b"):
        _ = embedder.revision


@pytest.mark.parametrize("mutation", ["lost_state", "renamed_source", "missing_collection"])
def test_status_and_recovery_do_not_silently_replace_existing_index(backend, mutation):
    result = sync(**backend)
    if mutation == "lost_state":
        backend["state_path"].unlink()
        with pytest.raises(RuntimeError, match="without local state"):
            sync(**backend)
        assert backend["store"].aliases["tradar_research"] == result["collection"]
        assert sync(**backend, rebuild=True)["status"] == "rebuilt"
        return
    if mutation == "renamed_source":
        path = backend["root"] / "rr-001-relative-reversal.md"
        path.rename(path.with_name("renamed.md"))
    else:
        del backend["store"].collections[result["collection"]]
    assert status(**{key: backend[key] for key in ("store", "root", "state_path")})["status"] == "stale_or_dirty"


def test_verifier_allowlist_excludes_any_retrieval_diagnostics(backend):
    record = load_research_records(backend["root"])[0]
    record.update(score=.9, matched_chunks=[{"score": 1}], dense_score=.8, bm25_score=3, rank=1)

    class Client:
        def create(self, payload):
            evidence = json.loads(payload["input"])["research_record"]
            assert not {"score", "dense_score", "bm25_score", "rank", "matched_chunks", "sections"} & set(evidence)
            assert "# Research Question" in evidence["text"] and "# Caveats" in evidence["text"]
            return {"output_text": '{"supported": true, "reason": "direct support"}'}

    assert verify_candidates("research", [record], client=Client())["status"] == "ok"


def test_extended_oracle_uses_existing_resource_envelope():
    from agent.core.resources import load_json
    from agent.retrieval.eval import run_eval
    from agent.retrieval.lexical_retriever import retrieve
    extension = load_json("eval/rag_retrieval.json")
    assert len(extension) == 6
    assert {case["slice"] for case in extension} == {"exact_identifier", "metadata_filter", "deleted_record", "superseded_record", "chunk_deduplication"}
    result = run_eval(cases=list(CASES) + list(extension), retriever=lambda query, **filters: retrieve(query, **{"include_superseded": False, **filters}))
    assert len(result["cases"]) == 29
    assert result["macro"]["error_count"] == 0


@pytest.mark.parametrize("mode", ["lexical", "bm25", "dense", "hybrid", "hybrid+verification"])
def test_filtered_cost_oracle_matches_canonical_metadata_and_wrong_market_abstains(backend, monkeypatch, mode):
    from agent.core.resources import load_json
    from agent.retrieval.lexical_retriever import retrieve
    from agent.retrieval.qdrant_store import FILTER_FIELDS, canonical_filter_payload
    from agent.retrieval.relevance_verifier_eval import FixtureClient

    records = load_research_records(backend["root"])
    canonical = {record["research_id"]: record for record in records}
    cases = {case["id"]: case for case in load_json("eval/rag_retrieval.json")}
    assert cases["rag_filtered_cost"]["market"] == canonical["RR-010"]["metadata"]["market"] == "multi-asset"
    monkeypatch.setattr("agent.retrieval.lexical_retriever.load_research_records", lambda: records)
    if mode != "lexical":
        sync(**backend)
        retriever = KnowledgeRetriever(**backend)
    for case_id in ("rag_filtered_cost", "rag_wrong_market"):
        case = cases[case_id]
        filters = {key: value for key, value in case.items() if key in FILTER_FIELDS}
        for research_id in case["relevant_ids"]:
            assert matches_filter(canonical_filter_payload(canonical[research_id], is_superseded=False), filter_spec(filters))
        if mode == "lexical":
            results = retrieve(case["query"], **filters)
        else:
            results = retriever.retrieve(case["query"], mode="hybrid" if mode == "hybrid+verification" else mode, **filters)
            if mode == "hybrid+verification":
                outcome = verify_candidates(case["query"], results, client=FixtureClient({"expected_supported": True}))
                assert outcome["status"] == ("ok" if case["relevant_ids"] else "abstain")
                results = outcome["results"]
        assert [record["research_id"] for record in results] == case["relevant_ids"]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("response", ["empty", "wrong_point"])
def test_each_retrieval_branch_must_prove_results_before_publication(backend, mode, response):
    old = sync(**backend)["collection"]
    original = backend["store"].query
    backend["store"].calls.clear()

    def broken_query(collection, request):
        selected = "hybrid" if "prefetch" in request else request["using"]
        return ([] if response == "empty" else [{"id": "unrelated"}]) if selected == mode else original(collection, request)

    backend["store"].query = broken_query
    with pytest.raises(RuntimeError, match=f"{mode} retrieval proof failed"):
        sync(**backend, rebuild=True)
    assert backend["store"].aliases["tradar_research"] == old
    assert not any(call[0] == "alias" for call in backend["store"].calls)
    assert not read_manifest(backend["state_path"])["ready"]


def test_sync_reconciles_existing_collection_after_creation_failure(backend):
    original = backend["store"].ensure_collection

    def interrupted(collection, profile):
        original(collection, profile)
        raise RuntimeError("payload indexes interrupted")

    backend["store"].ensure_collection = interrupted
    with pytest.raises(RuntimeError, match="indexes interrupted"):
        sync(**backend)
    pending = read_manifest(backend["state_path"])
    assert not pending["ready"] and backend["store"].exists(pending["collection"])
    recovery = Mock(side_effect=original)
    backend["store"].ensure_collection = recovery
    assert sync(**backend)["status"] == "synced"
    recovery.assert_called_once()
    assert read_manifest(backend["state_path"])["ready"]


def test_publication_rejects_filter_metadata_corruption(backend):
    original = backend["store"].upsert

    def tamper(collection, points):
        original(collection, points)
        next(iter(backend["store"].collections[collection].values()))["payload"]["market"] = "US-equity"

    backend["store"].upsert = tamper
    with pytest.raises(RuntimeError, match="filter metadata mismatch"):
        sync(**backend)
    assert not backend["store"].aliases and not read_manifest(backend["state_path"])["ready"]


@pytest.mark.parametrize("vector", [{"indices": [], "values": []}, {"indices": [1], "values": []},
                                    {"indices": [1], "values": [float("nan")]}, {"indices": [True], "values": [1.]},
                                    {"indices": [1, 1], "values": [1., 1.]}, {"indices": [1], "values": [0.]}])
def test_invalid_sparse_vector_cannot_publish(backend, vector):
    original = backend["store"].points

    def broken_points(collection, ids):
        points = original(collection, ids)
        points[0]["vector"]["bm25"] = vector
        return points

    backend["store"].points = broken_points
    with pytest.raises(RuntimeError, match="native BM25 vector missing/invalid"):
        sync(**backend)
    assert not backend["store"].aliases and not read_manifest(backend["state_path"])["ready"]


@pytest.mark.parametrize("previous_index", [False, True])
def test_empty_corpus_does_not_publish_or_delete_points(backend, previous_index):
    if previous_index:
        sync(**backend)
    before = deepcopy(backend["store"].collections), deepcopy(backend["store"].aliases)
    for path in backend["root"].glob("*.md"):
        path.unlink()
    backend["store"].calls.clear()
    backend["embedder"].inputs.clear()
    for rebuild in (False, True):
        with pytest.raises(ValueError, match="empty canonical corpus"):
            sync(**backend, rebuild=rebuild)
    assert before == (backend["store"].collections, backend["store"].aliases)
    assert not backend["store"].calls and not backend["embedder"].inputs
    assert backend["state_path"].exists() == previous_index
    if previous_index:
        with pytest.raises(RuntimeError, match="stale knowledge index"):
            KnowledgeRetriever(**backend).retrieve("research")


@pytest.mark.parametrize("field,value", [("market", "US-equity"), ("topic", "changed"), ("status", "rejected"),
                                        ("tags", ["other"]), ("research_id", "other"), ("date", "2020-01-01T00:00:00Z"),
                                        ("is_superseded", True), ("is_superseded", 0)])
def test_hydration_rejects_filter_payload_mismatch(backend, field, value):
    collection = sync(**backend)["collection"]
    point = next(iter(backend["store"].collections[collection].values()))
    point["payload"][field] = value
    with pytest.raises(RuntimeError, match="payload/source mismatch"):
        KnowledgeRetriever(**backend).retrieve("research", include_superseded=True)


def test_hydration_rejects_server_ignoring_canonical_filter(backend):
    collection = sync(**backend)["collection"]
    point = deepcopy(next(iter(backend["store"].collections[collection].values())))
    backend["store"].query = lambda collection, request: [{**point, "score": 1.0}]
    with pytest.raises(RuntimeError, match="violated canonical filters"):
        KnowledgeRetriever(**backend).retrieve("research", market="US-equity")


@pytest.mark.parametrize("cycle", ["self", "two_records"])
def test_supersession_cycles_abort_authoritative_scan(backend, cycle):
    _edit(backend, "supersedes: []", f"supersedes: [{'RR-001' if cycle == 'self' else 'RR-010'}]")
    if cycle == "two_records":
        _edit(backend, "supersedes: []", "supersedes: [RR-001]", "rr-010-trading-cost-defaults.md")
    with pytest.raises(ValueError, match="supersession cycle"):
        sync(**backend)
    assert not backend["store"].calls and not backend["state_path"].exists()


@pytest.mark.parametrize("metadata", ["extra: !!set {term: null}", "extra: 2026-01-01", "extra: {nested: []}",
                                      "tags: !!set {term: null}", "tags: [{nested: value}]", "date: 2026-01-01T00:00:00Z"])
def test_unsupported_yaml_metadata_shapes_abort_scan(backend, metadata):
    key = metadata.split(":")[0]
    if key == "extra":
        _edit(backend, "research_id: RR-001", metadata + "\nresearch_id: RR-001")
    elif key == "tags":
        _edit(backend, "tags: [relative-strength, short-term-reversal, cross-sectional, factor]", metadata)
    else:
        _edit(backend, "date: 2026-08-31", metadata)
    with pytest.raises(ValueError, match="unsupported frontmatter|tags must|date must"):
        sync(**backend)
    assert not backend["store"].calls and not backend["state_path"].exists()


def test_deleted_historical_supersession_and_nested_fence_are_valid(backend):
    _edit(backend, "supersedes: []", "supersedes: [RR-DELETED]")
    block = "````markdown\n```python\n# Not a record section\n```\n````"
    _edit(backend, "# Method", "# Method\n\n" + block)
    record = load_research_records(backend["root"])[0]
    assert block in record["sections"]["Method"]
    assert block in chunk_record(record, backend["profile"])[0]["text"]
    assert sync(**backend)["status"] == "synced"


def test_eval_error_rates_use_valid_query_denominator(capsys):
    from agent.retrieval.eval import run_eval, print_report
    cases = [{"id": name, "query": name, "relevant_ids": []} for name in ("failure", "false_positive", "abstain")]
    cases += [{"id": "positive_failure", "query": "failure", "relevant_ids": ["RR-001"]}]

    def retrieve(query, **filters):
        if query == "failure":
            return {"status": "error", "results": [], "errors": ["verifier failure"]}
        return [] if query == "abstain" else [{"research_id": "RR-001"}]

    result = run_eval(cases=cases, retriever=retrieve)
    assert result["macro"]["valid_query_count"] == result["macro"]["error_count"] == 2
    assert result["macro"]["by_k"] is None
    negative = result["macro"]["no_relevance"]
    assert negative["count"] == 3 and negative["valid_count"] == 2 and negative["error_count"] == 1
    assert negative["false_positive_rate"] == negative["abstention_rate"] == .5
    assert negative["empty_result_count"] == 1
    print_report("mixed", result)
    failed = run_eval(cases=cases[:1], retriever=retrieve)
    print_report("failed", failed)
    assert "not measured" in capsys.readouterr().out


@pytest.mark.parametrize("json_output", [False, True])
def test_qdrant_eval_cli_summary_and_full_json_preserve_results(monkeypatch, capsys, json_output):
    from agent.retrieval import eval as evaluation
    cases = [{"id": "positive", "query": "research", "relevant_ids": ["RR-001"]},
             {"id": "negative", "query": "none", "relevant_ids": []}]
    result = evaluation.run_eval(cases=cases, retriever=lambda query, **filters: [{"research_id": "RR-001"}] if query == "research" else [])
    result["macro"]["mean_latency_ms"] = {"eval_total_ms": 126.75, "embedding_ms": 100.0}
    modes = ("lexical", "bm25", "dense", "hybrid", "hybrid+verification")
    evaluations = {mode: deepcopy(result) for mode in modes}
    evaluations["hybrid+verification"] = evaluation.run_eval(cases=cases, retriever=lambda query, **filters: {"status": "error", "results": [], "errors": ["provider failed"]})
    original = deepcopy(evaluations)

    def compare(retrievers, *, cases):
        assert tuple(retrievers) == modes
        assert len(cases) == 29
        return evaluations

    monkeypatch.setattr(evaluation, "compare_retrievers", compare)
    monkeypatch.setattr("agent.retrieval.hybrid_retriever.KnowledgeRetriever", lambda: SimpleNamespace(search=Mock(side_effect=AssertionError("reporting test must not call services"))))
    monkeypatch.setattr("sys.argv", ["eval", "--qdrant", "--verification-provider", "fixture"] + (["--json"] if json_output else []))
    evaluation.main()
    output = capsys.readouterr().out
    assert evaluations == original
    if json_output:
        assert json.loads(output) == {"verification_provider": "fixture", "fixture_is_quality_measurement": False, "evaluations": original}
    else:
        lines = output.splitlines()
        rows = [line for line in lines if line.split()[0] in modes]
        assert [line.split()[0] for line in rows] == list(modes)
        assert len(lines) <= 10
        assert all(len(line) <= 120 for line in lines)
        assert "Recall@1/3/5" in output and "Precision@1/3/5" in output and "MRR@5" in output
        assert "FP rate" in output and "Abstain" in output and "Mean ms" in output and "Errors" in output
        assert "1.000/1.000/1.000" in rows[0] and "126.8" in rows[0]
        assert rows[-1].endswith("2/2") and "-/-/-" in rows[-1]
        assert "not model-quality measurement" in output and "--json" in output
        assert "provider failed" not in output and '"cases"' not in output


def test_qdrant_eval_oracle_accepts_batch_and_preserves_comparative_metrics(monkeypatch, capsys):
    from agent.retrieval import eval as evaluation
    records = [{"research_id": identity, "text": "full canonical fixture"} for identity in ("RR-001", "RR-010")]
    cases = [{"id": "cost", "query": "recorded costs", "relevant_ids": ["RR-010"]},
             {"id": "unknown", "query": "unrelated", "relevant_ids": []}]
    monkeypatch.setattr(evaluation, "CASES", cases)
    monkeypatch.setattr(evaluation, "load_json", lambda path: [])
    monkeypatch.setattr(evaluation, "retrieve", lambda query, **filters: records)
    monkeypatch.setattr("agent.retrieval.hybrid_retriever.KnowledgeRetriever", lambda: SimpleNamespace(search=lambda query, **filters: {"results": records, "latency_ms": {}}))
    monkeypatch.setattr("sys.argv", ["eval", "--qdrant", "--verification-provider", "fixture", "--json"])
    evaluation.main()
    result = json.loads(capsys.readouterr().out)["evaluations"]
    verified = result["hybrid+verification"]
    assert [row["top_results"] for row in verified["cases"]] == [["RR-010"], []]
    assert verified["macro"]["error_count"] == 0
    assert verified["macro"]["mrr@5"] == 1
    assert verified["macro"]["no_relevance"]["abstention_rate"] == 1
    assert all(result[mode]["macro"]["mrr@5"] == .5 for mode in ("lexical", "bm25", "dense", "hybrid"))
