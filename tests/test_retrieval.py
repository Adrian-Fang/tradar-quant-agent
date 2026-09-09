from __future__ import annotations

import unittest

from agent.core.resources import load_json
from agent.retrieval import load_research_records, retrieve


CASES = load_json("eval/retrieval.json")


class RetrievalTests(unittest.TestCase):
    def test_loader_reads_all_records_and_sections(self):
        records = load_research_records()
        self.assertEqual(len(records), 10)
        self.assertEqual(
            {record["research_id"] for record in records},
            {f"RR-{index:03d}" for index in range(1, 11)},
        )
        for record in records:
            self.assertTrue(record["path"].startswith("resources/knowledge/research/"))
            self.assertTrue(record["metadata"]["date"])
            self.assertTrue(record["question"])
            self.assertTrue(record["text"])
            self.assertEqual(record["source"], "slack")
            self.assertTrue(record["provenance"])

    def test_fixture_cases_return_expected_ids(self):
        self.assertEqual(len(CASES), 7)
        for case in CASES:
            results = retrieve(
                case["query"],
                market=case.get("market"),
                topic=case.get("topic"),
                status=case.get("status"),
                limit=case.get("limit", 5),
            )
            self.assertEqual(
                [result["research_id"] for result in results],
                case["expected_research_ids"],
                case["id"],
            )

    def test_results_include_score_metadata_text_and_provenance(self):
        result = retrieve("52-week high", limit=1)[0]
        self.assertGreater(result["score"], 0)
        self.assertEqual(result["metadata"]["research_id"], "RR-002")
        self.assertIn("HIGH52", result["text"])
        self.assertEqual(result["source"], "slack")
        self.assertTrue(result["source_ref"])
        self.assertTrue(result["provenance"])

    def test_limit_and_deterministic_order(self):
        first = retrieve("breakout", limit=2)
        second = retrieve("breakout", limit=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_status_filter_does_not_apply_by_default(self):
        all_results = retrieve("momentum")
        rejected_results = retrieve("momentum", status="rejected")
        self.assertTrue(all_results)
        self.assertTrue(rejected_results)
        self.assertTrue(all(result["metadata"]["status"] == "rejected" for result in rejected_results))


if __name__ == "__main__":
    unittest.main()
