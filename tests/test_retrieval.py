from __future__ import annotations

import unittest

from agent.core.resources import load_json
from agent.retrieval import load_research_records, retrieve
from agent.retrieval.eval import run_eval, score_case
from agent.retrieval.retrieval import _tokens


CASES = load_json("eval/retrieval.json")


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

    def test_score_case_math_for_multi_relevant_query(self):
        case = {"id": "multi", "relevant_ids": ["RR-A", "RR-B"]}
        row = score_case(case, ["RR-X", "RR-B", "RR-A"], (1, 3))
        self.assertEqual(row["hit@1"], 0.0)
        self.assertEqual(row["recall@1"], 0.0)
        self.assertEqual(row["precision@1"], 0.0)
        self.assertEqual(row["hit@3"], 1.0)
        self.assertEqual(row["recall@3"], 1.0)
        self.assertAlmostEqual(row["precision@3"], 2 / 3)
        self.assertEqual(row["reciprocal_rank"], 0.5)

    def test_no_relevance_case_is_separate_from_relevance_metrics(self):
        row = score_case(
            {"id": "none", "relevant_ids": []},
            ["RR-X"],
            (1, 3),
        )
        self.assertIsNone(row["hit@1"])
        self.assertIsNone(row["recall@1"])
        self.assertIsNone(row["precision@1"])
        self.assertEqual(row["reciprocal_rank"], 0.0)
        self.assertTrue(row["false_positive"])

        empty = score_case({"id": "none", "relevant_ids": []}, [], (1,))
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

    def test_limit_is_applied_without_metric_quality_assumption(self):
        self.assertEqual(len(retrieve("breakout", limit=2)), 2)


if __name__ == "__main__":
    unittest.main()
