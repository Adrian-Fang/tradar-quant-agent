from __future__ import annotations

import json
import unittest
from copy import deepcopy

import pytest

from agent.retrieval.semantic_retriever import prepare_semantic_corpus
from agent.retrieval.relevance_verifier import retrieve_verified, verify_candidates, verify_record


RECORDS = (
    {
        "research_id": "RR-A",
        "path": "resources/knowledge/research/rr-a.md",
        "metadata": {"market": "A-share", "topic": "a", "status": "validated"},
        "title": "A",
        "question": "question A",
        "tags": [],
        "text": "body A",
        "source": "research_record",
        "source_ref": "source-a",
        "provenance": "provenance-a",
    },
    {
        "research_id": "RR-B",
        "path": "resources/knowledge/research/rr-b.md",
        "metadata": {"market": "B-share", "topic": "b", "status": "validated"},
        "title": "B",
        "question": "question B",
        "tags": [],
        "text": "body B",
        "source": "research_record",
        "source_ref": "source-b",
        "provenance": "provenance-b",
    },
    {
        "research_id": "RR-C",
        "path": "resources/knowledge/research/rr-c.md",
        "metadata": {"market": "B-share", "topic": "c", "status": "rejected"},
        "title": "C",
        "question": "question C",
        "tags": [],
        "text": "body C",
        "source": "research_record",
        "source_ref": "source-c",
        "provenance": "provenance-c",
    },
)


class FakeEmbedder:
    def __init__(self, document_vectors):
        self.document_vectors = document_vectors
        self.calls = []

    def __call__(self, texts):
        self.calls.append(texts)
        if len(texts) == len(RECORDS):
            return self.document_vectors
        return [[1.0, 0.0]]


class FakeVerifierClient:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []

    def create(self, payload):
        body = json.loads(payload["input"])
        self.calls.append(body)
        if "research_records" in body:
            return {"output_text": json.dumps({"results": [
                {"research_id": record["research_id"], "supported": self.decisions[record["research_id"]], "reason": "fake verifier result"}
                for record in body["research_records"]
            ]})}
        research_id = body["research_record"]["research_id"]
        return {
            "output_text": json.dumps({
                "supported": self.decisions[research_id],
                "reason": "fake verifier result",
            })
        }


def prepared(vectors):
    embedder = FakeEmbedder(vectors)
    corpus = prepare_semantic_corpus(embedder=embedder, records=RECORDS)
    return corpus, embedder


class RetrievalRuntimeTests(unittest.TestCase):
    def test_accepts_positive_and_rejects_near_miss_in_semantic_order(self):
        corpus, embedder = prepared([[1, 0], [0.8, 0.6], [0, 1]])
        verifier = FakeVerifierClient({"RR-A": True, "RR-B": False})

        result = retrieve_verified(
            "query",
            client=verifier,
            candidate_limit=2,
            embedder=embedder,
            prepared_corpus=corpus,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual([item["research_id"] for item in result["results"]], ["RR-A"])
        self.assertEqual([item["research_id"] for item in result["rejected"]], ["RR-B"])

    def test_verifier_receives_record_without_score_but_result_keeps_score(self):
        corpus, embedder = prepared([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
        verifier = FakeVerifierClient({"RR-A": True})

        result = retrieve_verified(
            "query",
            client=verifier,
            candidate_limit=1,
            embedder=embedder,
            prepared_corpus=corpus,
        )

        self.assertNotIn("score", verifier.calls[0]["research_record"])
        self.assertEqual(result["results"][0]["score"], 1.0)

    def test_keeps_supported_results_in_semantic_order(self):
        corpus, embedder = prepared([[0.8, 0.6], [1, 0], [0, 1]])
        verifier = FakeVerifierClient({"RR-A": True, "RR-B": True})

        result = retrieve_verified(
            "query",
            client=verifier,
            candidate_limit=2,
            embedder=embedder,
            prepared_corpus=corpus,
        )

        self.assertEqual([item["research_id"] for item in result["results"]], ["RR-B", "RR-A"])

    def test_all_rejected_returns_abstain(self):
        corpus, embedder = prepared([[1, 0], [0.8, 0.6], [0, 1]])
        verifier = FakeVerifierClient({"RR-A": False, "RR-B": False})

        result = retrieve_verified(
            "query",
            client=verifier,
            candidate_limit=2,
            embedder=embedder,
            prepared_corpus=corpus,
        )

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["results"], [])
        self.assertEqual([item["research_id"] for item in result["rejected"]], ["RR-A", "RR-B"])
        self.assertEqual(result["errors"], [])

    def test_provider_and_malformed_errors_fail_closed_separately(self):
        corpus, embedder = prepared([[1, 0], [0, 1], [0, 1]])

        class ProviderErrorClient:
            def create(self, payload):
                raise RuntimeError("offline")

        provider_error = retrieve_verified(
            "query",
            client=ProviderErrorClient(),
            candidate_limit=1,
            embedder=embedder,
            prepared_corpus=corpus,
        )
        self.assertEqual(provider_error["status"], "error")
        self.assertEqual(provider_error["results"], [])
        self.assertEqual(provider_error["errors"][0]["error_type"], "provider_error")

        class MalformedClient:
            def create(self, payload):
                return {"output_text": "not json"}

        malformed = retrieve_verified(
            "query",
            client=MalformedClient(),
            candidate_limit=1,
            embedder=embedder,
            prepared_corpus=corpus,
        )
        self.assertEqual(malformed["status"], "error")
        self.assertEqual(malformed["results"], [])
        self.assertEqual(malformed["errors"][0]["error_type"], "malformed_response")

    def test_partial_batch_and_provider_errors_fail_closed(self):
        corpus, embedder = prepared([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])

        class LaterErrorClient:
            def __init__(self, malformed):
                self.malformed = malformed
                self.calls = []

            def create(self, payload):
                body = json.loads(payload["input"])
                self.calls.append([record["research_id"] for record in body["research_records"]])
                if self.malformed:
                    # One supported row followed by an invalid row is atomic failure.
                    return {"output_text": json.dumps({"results": [
                        {"research_id": "RR-A", "supported": True, "reason": "accepted"},
                        {"research_id": "RR-B", "supported": "yes", "reason": "invalid"},
                    ]})}
                raise RuntimeError("offline")

        for malformed, error_type in ((False, "provider_error"), (True, "malformed_response")):
            with self.subTest(error_type=error_type):
                verifier = LaterErrorClient(malformed)
                result = retrieve_verified(
                    "query",
                    client=verifier,
                    candidate_limit=2,
                    embedder=embedder,
                    prepared_corpus=corpus,
                )

                self.assertEqual(verifier.calls, [["RR-A", "RR-B"]])
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["results"], [])
                self.assertEqual([error["research_id"] for error in result["errors"]], ["RR-A", "RR-B"])
                self.assertEqual(result["errors"][0]["error_type"], error_type)

    def test_metadata_filter_reaches_semantic_retrieval_before_verification(self):
        corpus, embedder = prepared([[1, 0], [0.8, 0.6], [0, 1]])
        verifier = FakeVerifierClient({"RR-A": True})

        result = retrieve_verified(
            "query",
            client=verifier,
            market="A-share",
            candidate_limit=3,
            embedder=embedder,
            prepared_corpus=corpus,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual([body["research_record"]["research_id"] for body in verifier.calls], ["RR-A"])


DECISIONS = [
    {"research_id": "RR-A", "supported": True, "reason": "A evidence"},
    {"research_id": "RR-B", "supported": False, "reason": "B mismatch"},
]


@pytest.mark.parametrize("value", [
    "not json", [], {}, {"results": {}}, {"results": []},
    {"results": [DECISIONS[0]]},
    {"results": [*DECISIONS, DECISIONS[0]]},
    {"results": [DECISIONS[0], {**DECISIONS[1], "research_id": "RR-UNKNOWN"}]},
    {"results": [DECISIONS[0], {**DECISIONS[1], "research_id": []}]},
    {"results": [DECISIONS[0], {**DECISIONS[1], "supported": "false"}]},
    {"results": [DECISIONS[0], {**DECISIONS[1], "supported": 1}]},
    {"results": [DECISIONS[0], {**DECISIONS[1], "reason": None}]},
    {"results": [DECISIONS[0], {**DECISIONS[1], "score": 1}]},
    {"results": [DECISIONS[0], None]},
    {"results": DECISIONS, "extra": True},
])
def test_batch_malformed_decisions_fail_atomically(value):
    class Client:
        calls = 0

        def create(self, payload):
            self.calls += 1
            return {"output_text": value if isinstance(value, str) else json.dumps(value)}

    client = Client()
    result = verify_candidates("query", list(RECORDS[:2]), client=client)
    assert client.calls == 1
    assert result["status"] == "error" and result["results"] == []
    assert [error["research_id"] for error in result["errors"]] == ["RR-A", "RR-B"]
    assert all(error["error_type"] == "malformed_response" and error["error"] for error in result["errors"])


def test_batch_reordered_response_preserves_input_order_and_per_record_reasons():
    records = [deepcopy(record) for record in RECORDS]
    for record in records:
        record.update(score=1, matched_chunks=[{}], retrieval_debug="diagnostic only")

    class Client:
        calls = []

        def create(self, payload):
            self.calls.append(payload)
            return {"output_text": json.dumps({"results": [
                {"research_id": "RR-C", "supported": True, "reason": "C evidence"},
                *reversed(DECISIONS),
            ]})}

    client = Client()
    result = verify_candidates("query", records, client=client)
    assert len(client.calls) == 1
    assert [(record["research_id"], record["verification"]["reason"]) for record in result["results"]] == [("RR-A", "A evidence"), ("RR-C", "C evidence")]
    assert result["rejected"] == [{"research_id": "RR-B", "reason": "B mismatch"}]
    body = json.loads(client.calls[0]["input"])
    assert set(body) == {"query", "research_records"}
    assert [record["text"] for record in body["research_records"]] == [record["text"] for record in records]
    assert all(not (set(record) & {"score", "matched_chunks", "retrieval_debug"}) for record in body["research_records"])
    assert "Never pool evidence across records" in client.calls[0]["instructions"]


@pytest.mark.parametrize("identity", [None, "", [], "RR-A"])
def test_invalid_batch_candidate_identity_fails_before_provider(identity):
    class Client:
        def create(self, payload):
            raise AssertionError("must not call provider")

    result = verify_candidates("query", [RECORDS[0], {**RECORDS[1], "research_id": identity}], client=Client())
    assert result["status"] == "error" and result["results"] == []
    assert all(error["error_type"] == "invalid_candidates" for error in result["errors"])


def test_batch_and_single_verification_preserve_labeled_eval_decisions():
    from agent.retrieval.loader import load_research_records
    from agent.retrieval.relevance_verifier_eval import CASES

    records = load_research_records()
    for case in CASES:
        candidates = sorted(records, key=lambda record: record["research_id"] != case["research_id"])[:5]
        decisions = {record["research_id"]: case["expected_supported"] if record["research_id"] == case["research_id"] else False for record in candidates}
        client = FakeVerifierClient(decisions)
        single = [verify_record(case["query"], record, client=client) for record in candidates]
        batch_client = FakeVerifierClient(decisions)
        batch = verify_candidates(case["query"], candidates, client=batch_client)
        assert [record["research_id"] for record in batch["results"]] == [record["research_id"] for record, decision in zip(candidates, single) if decision["supported"]]
        assert [row["research_id"] for row in batch["rejected"]] == [record["research_id"] for record, decision in zip(candidates, single) if not decision["supported"]]
        assert batch["status"] == ("ok" if case["expected_supported"] else "abstain")
        assert len(client.calls) == 5 and len(batch_client.calls) == 1
        assert "expected_supported" not in json.dumps(batch_client.calls)


if __name__ == "__main__":
    unittest.main()
