from __future__ import annotations

import json
import unittest

from agent.retrieval.loader import load_research_records
from agent.retrieval.verification_eval import (
    CASES,
    _prompt,
    run_case,
    run_eval,
    score_case,
)


class RetrievalVerificationTests(unittest.TestCase):
    def test_dataset_schema_and_balanced_slices(self):
        self.assertEqual(len(CASES), 20)
        self.assertEqual(
            {case["slice"] for case in CASES},
            {"positive", "negative"},
        )
        self.assertEqual(sum(case["expected_supported"] for case in CASES), 10)
        self.assertEqual(sum(not case["expected_supported"] for case in CASES), 10)
        self.assertEqual(
            {case["research_id"] for case in CASES if case["expected_supported"]},
            {"RR-001", "RR-002", "RR-003", "RR-004", "RR-005", "RR-006", "RR-007", "RR-008", "RR-009", "RR-010"},
        )
        for case in CASES:
            self.assertIsInstance(case["id"], str)
            self.assertIsInstance(case["query"], str)
            self.assertIsInstance(case["research_id"], str)
            self.assertIsInstance(case["expected_supported"], bool)

    def test_prompt_contains_record_and_no_case_oracle_fields(self):
        case = CASES[0]
        record = next(
            record for record in load_research_records()
            if record["research_id"] == case["research_id"]
        )
        payload = _prompt(case, record)
        body = json.loads(payload["input"])
        self.assertEqual(set(body), {"query", "research_record"})
        self.assertIn("research_id", body["research_record"])
        serialized = json.dumps(body, ensure_ascii=False)
        for oracle_field in ("expected_supported", "slice", "difficulty"):
            self.assertNotIn(oracle_field, serialized)
        self.assertNotIn("expected_supported", payload["instructions"])

    def test_positive_and_negative_scoring(self):
        outcome = {
            "parsed": {"supported": True, "reason": "evidence"},
            "parse_error": None,
            "structure_error": None,
            "response_text": "{}",
        }
        positive = next(case for case in CASES if case["expected_supported"])
        self.assertEqual(score_case(positive, outcome, 1)["status"], "pass")
        self.assertEqual(
            score_case(
                positive,
                {**outcome, "parsed": {"supported": False, "reason": "no"}},
                1,
            )["status"],
            "fail",
        )

        negative = next(case for case in CASES if not case["expected_supported"])
        self.assertEqual(
            score_case(
                negative,
                {**outcome, "parsed": {"supported": False, "reason": "no"}},
                1,
            )["status"],
            "pass",
        )

    def test_fixture_passes_all_cases_and_reports_negative_slices(self):
        rows, meta = run_eval("fixture", repeats=1)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["status"] == "pass" for row in rows))
        self.assertEqual(meta["slice_metrics"]["positive"]["positive_acceptance"], 1.0)
        negative = meta["slice_metrics"]["negative"]
        self.assertEqual(negative["negative_rejection"], 1.0)
        self.assertEqual(negative["hard_negative_rejection"], 1.0)
        self.assertEqual(negative["near_miss_negative_rejection"], 1.0)

    def test_malformed_response_and_provider_error_are_human_review(self):
        case = CASES[0]
        record = next(
            record for record in load_research_records()
            if record["research_id"] == case["research_id"]
        )

        class MalformedClient:
            def create(self, payload):
                return {"output_text": "not json"}

        malformed = score_case(
            case,
            run_case(case, record, client=MalformedClient(), model="fixture"),
            1,
        )
        self.assertEqual(malformed["status"], "human_review")
        self.assertEqual(malformed["failure_type"], "malformed_response")

        class ProviderErrorClient:
            def create(self, payload):
                raise RuntimeError("offline")

        provider_error = score_case(
            case,
            run_case(case, record, client=ProviderErrorClient(), model="fixture"),
            1,
        )
        self.assertEqual(provider_error["status"], "human_review")
        self.assertEqual(provider_error["failure_type"], "provider_error")


if __name__ == "__main__":
    unittest.main()
