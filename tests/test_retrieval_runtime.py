from __future__ import annotations

import json
import unittest

from agent.retrieval.semantic import prepare_semantic_corpus
from agent.retrieval.verification import retrieve_verified


RECORDS = (
    {
        "research_id": "RR-A",
        "path": "resources/knowledge/research/rr-a.md",
        "metadata": {"market": "A-share", "topic": "a", "status": "validated"},
        "title": "A",
        "question": "question A",
        "tags": [],
        "text": "body A",
        "source": "slack",
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
        "source": "slack",
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
        "source": "slack",
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
        research_id = body["research_record"]["research_id"]
        self.calls.append(body)
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

    def test_prior_supported_candidate_is_cleared_when_later_verification_errors(self):
        corpus, embedder = prepared([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])

        class LaterErrorClient:
            def __init__(self, malformed):
                self.malformed = malformed
                self.calls = []
                self.supported_calls = []

            def create(self, payload):
                body = json.loads(payload["input"])
                research_id = body["research_record"]["research_id"]
                self.calls.append(research_id)
                if research_id == "RR-A":
                    self.supported_calls.append(research_id)
                    return {"output_text": json.dumps({
                        "supported": True,
                        "reason": "accepted",
                    })}
                if self.malformed:
                    return {"output_text": "not json"}
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

                self.assertEqual(verifier.supported_calls, ["RR-A"])
                self.assertEqual(verifier.calls, ["RR-A", "RR-B"])
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["results"], [])
                self.assertEqual(result["errors"][0]["research_id"], "RR-B")
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


if __name__ == "__main__":
    unittest.main()
