from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.core.resources import load_json
from agent.core.providers import OllamaEmbeddingClient, OpenAIEmbeddingClient
from agent.retrieval import load_research_records, retrieve
from agent.retrieval.eval import run_eval, run_case
from agent.retrieval.semantic import prepare_semantic_corpus, retrieve_semantic
from agent.retrieval.retrieval import _tokens


CASES = load_json("eval/retrieval.json")
FAKE_RECORDS = (
    {
        "research_id": "RR-B",
        "path": "resources/knowledge/research/rr-b.md",
        "metadata": {"market": "A-share", "topic": "b", "status": "validated"},
        "title": "B",
        "question": "question B",
        "tags": [],
        "text": "body B",
        "source": "slack",
        "source_ref": "source-b",
        "provenance": "provenance-b",
    },
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


class RetrievalTests(unittest.TestCase):
    def test_tokens_keep_english_and_numbers_as_words(self):
        self.assertEqual(_tokens("52-Week HIGH"), {"52", "week", "high"})

    def test_tokens_use_chinese_character_bigrams(self):
        self.assertEqual(_tokens("相对强弱"), {"相对", "对强", "强弱"})

    def test_tokens_handle_mixed_text(self):
        self.assertEqual(
            _tokens("买入10bp卖出15bp"),
            {"买入", "卖出", "10bp", "15bp"},
        )

    def test_tokens_deduplicate_repeated_words_and_bigrams(self):
        self.assertEqual(_tokens("买入 买入 10bp 10bp"), {"买入", "10bp"})

    def test_loader_invariants(self):
        records = load_research_records()
        self.assertTrue(records)
        research_ids = [record["research_id"] for record in records]
        self.assertEqual(len(research_ids), len(set(research_ids)))
        for record in records:
            self.assertTrue(record["path"].startswith("resources/knowledge/research/"))
            self.assertTrue(record["metadata"]["research_id"])
            self.assertTrue(record["metadata"]["date"])
            self.assertTrue(record["metadata"]["status"])
            self.assertTrue(record["question"])
            self.assertTrue(record["text"])
            self.assertTrue(record["source"])
            self.assertTrue(record["provenance"])

    def test_relevance_dataset_has_oracle_not_exact_output(self):
        self.assertGreaterEqual(len(CASES), 10)
        for case in CASES:
            self.assertIn("id", case)
            self.assertIn("query", case)
            self.assertIn("relevant_ids", case)
            self.assertNotIn("expected_research_ids", case)
        self.assertEqual(
            {case.get("slice", "baseline") for case in CASES},
            {"baseline", "semantic_challenge"},
        )
        self.assertEqual(
            sum(case.get("slice") == "semantic_challenge" for case in CASES),
            5,
        )

    def test_score_case_math_for_multi_relevant_query(self):
        case = {"id": "multi", "relevant_ids": ["RR-A", "RR-B"]}
        row = run_case(case, ["RR-X", "RR-B", "RR-A"], (1, 3))
        self.assertEqual(row["hit@1"], 0.0)
        self.assertEqual(row["recall@1"], 0.0)
        self.assertEqual(row["precision@1"], 0.0)
        self.assertEqual(row["hit@3"], 1.0)
        self.assertEqual(row["recall@3"], 1.0)
        self.assertAlmostEqual(row["precision@3"], 2 / 3)
        self.assertEqual(row["reciprocal_rank"], 0.5)

    def test_no_relevance_case_is_separate_from_relevance_metrics(self):
        row = run_case(
            {"id": "none", "relevant_ids": []},
            ["RR-X"],
            (1, 3),
        )
        self.assertIsNone(row["hit@1"])
        self.assertIsNone(row["recall@1"])
        self.assertIsNone(row["precision@1"])
        self.assertEqual(row["reciprocal_rank"], 0.0)
        self.assertTrue(row["false_positive"])

        empty = run_case({"id": "none", "relevant_ids": []}, [], (1,))
        self.assertFalse(empty["false_positive"])

    def test_metadata_filter(self):
        results = retrieve(
            "transaction cost",
            market="multi-asset",
            status="validated",
        )
        self.assertTrue(results)
        self.assertTrue(
            all(
                result["metadata"]["market"] == "multi-asset"
                and result["metadata"]["status"] == "validated"
                for result in results
            )
        )

    def test_deterministic_eval_and_no_relevance_aggregate(self):
        first = run_eval()
        second = run_eval()
        self.assertEqual(first, second)
        self.assertEqual(first["k_values"], [1, 3, 5])
        no_relevance = first["macro"]["no_relevance"]
        self.assertEqual(no_relevance["count"], 1)
        self.assertEqual(
            no_relevance["false_positive_count"],
            sum(
                row["false_positive"]
                for row in first["cases"]
                if not row["relevant"]
            ),
        )
        self.assertIn("1", first["macro"]["by_k"])
        self.assertIn("5", first["macro"]["by_k"])

    def test_baseline_and_semantic_challenge_metrics_are_separate(self):
        evaluation = run_eval()
        self.assertEqual(
            set(evaluation["slices"]),
            {"baseline", "semantic_challenge"},
        )
        self.assertEqual(evaluation["slices"]["baseline"]["no_relevance"]["count"], 1)
        self.assertEqual(evaluation["slices"]["semantic_challenge"]["no_relevance"]["count"], 0)
        self.assertIn("mrr", evaluation["slices"]["semantic_challenge"])

    def test_limit_is_applied_without_metric_quality_assumption(self):
        self.assertEqual(len(retrieve("breakout", limit=2)), 2)

    def test_semantic_cosine_ranking_tie_break_and_limit(self):
        def embedder(texts):
            return [[1, 0], [0.8, 0.6], [1, 0], [0, 1]]

        with patch("agent.retrieval.semantic.load_research_records", return_value=FAKE_RECORDS):
            results = retrieve_semantic("query", limit=2, embedder=embedder)
        self.assertEqual([result["research_id"] for result in results], ["RR-A", "RR-B"])
        self.assertEqual(results[0]["score"], 1.0)

    def test_semantic_filter_happens_before_embedding_and_keeps_provenance(self):
        calls = []

        def embedder(texts):
            calls.append(texts)
            return [[1, 0], [1, 0], [1, 0]]

        with patch("agent.retrieval.semantic.load_research_records", return_value=FAKE_RECORDS):
            results = retrieve_semantic("query", market="A-share", embedder=embedder)
        self.assertEqual([result["research_id"] for result in results], ["RR-A", "RR-B"])
        self.assertEqual(len(calls[0]), 3)
        self.assertEqual(calls[0][1:], ["B\nbody B", "A\nbody A"])
        self.assertEqual(results[0]["source"], "slack")
        self.assertTrue(results[0]["provenance"])

    def test_prepared_semantic_eval_embeds_documents_once(self):
        calls = []

        def embedder(texts):
            calls.append(texts)
            return [[1, 0] for _ in texts]

        prepared_corpus = prepare_semantic_corpus(embedder=embedder)
        run_eval(
            retriever=lambda query, **filters: retrieve_semantic(
                query,
                embedder=embedder,
                prepared_corpus=prepared_corpus,
                **filters,
            )
        )

        self.assertEqual(len(calls), len(CASES) + 1)
        self.assertEqual(len(calls[0]), len(prepared_corpus["records"]))
        self.assertTrue(all(len(texts) == 1 for texts in calls[1:]))

    def test_openai_embedding_adapter_batches_in_core_provider(self):
        class FakeEmbeddings:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=[2.0]),
                        SimpleNamespace(index=0, embedding=[1.0]),
                    ]
                )

        sdk = SimpleNamespace(embeddings=FakeEmbeddings())
        client = OpenAIEmbeddingClient(
            api_key="test-key",
            model="test-model",
            sdk_client=sdk,
        )
        self.assertEqual(client(["query", "record"]), [[1.0], [2.0]])
        self.assertEqual(
            sdk.embeddings.calls,
            [{"model": "test-model", "input": ["query", "record"]}],
        )

    def test_ollama_embedding_adapter_batches_and_preserves_order(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"embeddings": [[1.0], [2.0]]}'

        calls = []

        def urlopen(http_request, timeout):
            calls.append((http_request, timeout))
            return FakeResponse()

        with patch("agent.core.providers.request.urlopen", side_effect=urlopen):
            client = OllamaEmbeddingClient(
                base_url="http://ollama/",
                model="qwen3-embedding:0.6b",
                timeout=3,
            )
            self.assertEqual(client(["query", "record"]), [[1.0], [2.0]])

        http_request, timeout = calls[0]
        self.assertEqual(http_request.full_url, "http://ollama/api/embed")
        self.assertEqual(timeout, 3)
        self.assertEqual(
            json.loads(http_request.data),
            {"model": "qwen3-embedding:0.6b", "input": ["query", "record"]},
        )

    def test_ollama_embedding_adapter_reports_transport_and_response_errors(self):
        client = OllamaEmbeddingClient()
        with patch(
            "agent.core.providers.request.urlopen",
            side_effect=OSError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Ollama embedding request failed"):
                client(["query"])

        class EmptyResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"{}"

        with patch("agent.core.providers.request.urlopen", return_value=EmptyResponse()):
            with self.assertRaisesRegex(RuntimeError, "invalid embeddings batch"):
                client(["query"])

    def test_semantic_empty_query_and_invalid_vectors(self):
        def unexpected_embedder(texts):
            raise AssertionError("empty query should not be embedded")

        with patch("agent.retrieval.semantic.load_research_records", return_value=FAKE_RECORDS):
            self.assertEqual(retrieve_semantic("  ", embedder=unexpected_embedder), [])

            with self.assertRaises(ValueError):
                retrieve_semantic("query", embedder=lambda texts: [[1, 0]])

    def test_eval_accepts_a_second_retriever_without_changing_cases(self):
        calls = []

        def fake_retriever(query, **filters):
            calls.append((query, filters))
            return []

        evaluation = run_eval(retriever=fake_retriever)
        self.assertEqual(len(calls), len(CASES))
        self.assertEqual(len(evaluation["cases"]), len(CASES))


if __name__ == "__main__":
    unittest.main()
