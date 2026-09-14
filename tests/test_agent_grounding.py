from __future__ import annotations

import json
import unittest

from agent.grounding.eval import (
    CASES,
    FixtureClient,
    _metrics,
    _prompt,
    run_case,
    run_eval,
    score_case,
)


def outcome(parsed):
    return {
        "response_text": json.dumps(parsed),
        "parsed": parsed,
        "parse_error": None,
        "structure_error": None,
    }


class GroundingEvalTests(unittest.TestCase):
    def test_dataset_schema_and_required_slices(self):
        self.assertGreaterEqual(len(CASES), 12)
        self.assertLessEqual(len(CASES), 16)
        self.assertTrue({case["slice"] for case in CASES} >= {
            "exactly_supported", "numeric_fidelity", "time_scope", "market_scope",
            "direction/polarity", "unsupported_extrapolation", "contradiction",
            "multi_evidence_synthesis", "insufficient_evidence",
        })
        for case in CASES:
            self.assertIsInstance(case["answer"], str)
            self.assertIsInstance(case["evidence"], list)
            evidence_ids = {item["id"] for item in case["evidence"]}
            self.assertTrue(evidence_ids)
            for expected in case["expected_claims"]:
                self.assertIn(expected["label"], {"supported", "unsupported", "contradicted", "unverifiable"})
                self.assertTrue(set(expected["evidence_ids"]) <= evidence_ids)

    def test_prompt_has_answer_and_evidence_without_oracle(self):
        case = CASES[0]
        payload = _prompt(case)
        body = json.loads(payload["input"])
        self.assertEqual(set(body), {"answer", "evidence"})
        serialized = json.dumps(body, ensure_ascii=False)
        for oracle in ("expected_claims", "expected_grounded", "slice", "difficulty"):
            self.assertNotIn(oracle, serialized)
        self.assertNotIn("expected_claims", payload["instructions"])

    def test_fixture_passes_all_cases(self):
        rows, meta = run_eval(provider="fixture", repeats=1)
        self.assertEqual(len(rows), len(CASES))
        self.assertTrue(all(row["case_pass"] for row in rows))
        self.assertEqual(meta["metrics"]["claim_label_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["evidence_attribution_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["answer_groundedness_accuracy"], 1.0)

    def test_supported_claim_and_multi_evidence(self):
        case = next(case for case in CASES if case["id"] == "multi_evidence_breakout_conclusion")
        parsed = {
            "answer": case["answer"],
            "claims": [{
                "claim": case["answer"],
                "evidence_ids": ["e1", "e2"],
                "grounding": "supported",
            }],
        }
        row = score_case(case, outcome(parsed), 1)
        self.assertTrue(row["case_pass"])
        self.assertEqual(row["evidence_attribution_accuracy"], 1.0)

    def test_split_claims_and_changed_claim_order_do_not_fail(self):
        case = next(case for case in CASES if case["id"] == "multi_evidence_breakout_conclusion")
        split = {
            "answer": case["answer"],
            "claims": [
                {"claim": "The execution result did not reliably transfer to next-open.", "evidence_ids": ["e2"], "grounding": "supported"},
                {"claim": "There was no stable incremental volume advantage.", "evidence_ids": ["e1"], "grounding": "supported"},
            ],
        }
        self.assertTrue(score_case(case, outcome(split), 1)["case_pass"])

        two_claim_oracle = {
            **case,
            "expected_claims": [
                {"label": "supported", "evidence_ids": ["e1"]},
                {"label": "supported", "evidence_ids": ["e2"]},
            ],
        }
        reordered = {
            "answer": case["answer"],
            "claims": [
                {"claim": "next-open was not reliable", "evidence_ids": ["e2"], "grounding": "supported"},
                {"claim": "no stable incremental advantage", "evidence_ids": ["e1"], "grounding": "supported"},
            ],
        }
        self.assertTrue(score_case(two_claim_oracle, outcome(reordered), 1)["case_pass"])

    def test_unsupported_and_contradicted_claims(self):
        for case_id, label in (
            ("unsupported_rs_extrapolation", "contradicted"),
            ("polarity_high52_positive_alpha", "contradicted"),
        ):
            case = next(case for case in CASES if case["id"] == case_id)
            parsed = {
                "answer": case["answer"],
                "claims": [{
                    "claim": case["answer"],
                    "evidence_ids": ["e1"],
                    "grounding": label,
                }],
            }
            self.assertTrue(score_case(case, outcome(parsed), 1)["case_pass"])

    def test_answer_groundedness_is_separate_from_label_accuracy(self):
        case = next(case for case in CASES if case["id"] == "time_scope_future_extrapolation")
        row = score_case(case, outcome({
            "answer": case["answer"],
            "claims": [{"claim": case["answer"], "evidence_ids": ["e1"], "grounding": "contradicted"}],
        }), 1)
        self.assertEqual(row["claim_label_accuracy"], 0.0)
        self.assertTrue(row["answer_groundedness_correct"])
        self.assertFalse(row["case_pass"])

    def test_label_confusion_and_per_label_recall(self):
        rows = []
        for case_id in ("time_scope_future_extrapolation", "insufficient_old_script_cost_coverage"):
            case = next(case for case in CASES if case["id"] == case_id)
            rows.append(score_case(case, outcome({
                "answer": case["answer"],
                "claims": [{"claim": case["answer"], "evidence_ids": ["e1"], "grounding": "contradicted"}],
            }), 1))

        metrics = _metrics(rows)
        self.assertEqual(metrics["label_confusion"]["unsupported"]["contradicted"], 1)
        self.assertEqual(metrics["label_confusion"]["unverifiable"]["contradicted"], 1)
        self.assertEqual(metrics["per_label_recall"]["unsupported"], 0.0)
        self.assertEqual(metrics["per_label_recall"]["unverifiable"], 0.0)

    def test_numeric_mismatch_and_scope_mismatch_are_not_supported(self):
        numeric = next(case for case in CASES if case["id"] == "numeric_sell_cost_mismatch")
        scope = next(case for case in CASES if case["id"] == "market_scope_global_breakout")
        for case in (numeric, scope):
            parsed = {
                "answer": case["answer"],
                "claims": [{"claim": case["answer"], "evidence_ids": ["e1"], "grounding": case["expected_claims"][0]["label"]}],
            }
            row = score_case(case, outcome(parsed), 1)
            self.assertTrue(row["case_pass"])

    def test_insufficient_evidence_label_passes(self):
        case = next(case for case in CASES if case["slice"] == "insufficient_evidence")
        parsed = {
            "answer": case["answer"],
            "claims": [{"claim": case["answer"], "evidence_ids": ["e1"], "grounding": "unverifiable"}],
        }
        self.assertTrue(score_case(case, outcome(parsed), 1)["case_pass"])

    def test_invalid_evidence_id_and_missing_citation_are_contract_failures(self):
        case = CASES[0]
        class InvalidClient:
            def __init__(self, evidence_ids):
                self.evidence_ids = evidence_ids

            def create(self, payload):
                return {"output_text": json.dumps({
                    "answer": case["answer"],
                    "claims": [{
                        "claim": case["answer"],
                        "evidence_ids": self.evidence_ids,
                        "grounding": "supported",
                    }],
                })}

        invalid = score_case(
            case,
            run_case(case, client=InvalidClient(["missing"]), model="fixture"),
            1,
        )
        missing = score_case(
            case,
            run_case(case, client=InvalidClient([]), model="fixture"),
            1,
        )
        self.assertEqual(invalid["eval_status"], "human_review")
        self.assertEqual(invalid["failure_type"], "contract_violation")
        self.assertEqual(missing["eval_status"], "human_review")
        self.assertEqual(missing["failure_type"], "contract_violation")

    def test_malformed_response_is_human_review(self):
        case = CASES[0]

        class MalformedClient:
            def create(self, payload):
                return {"output_text": "not json"}

        row = score_case(case, run_case(case, client=MalformedClient(), model="fixture"), 1)
        self.assertEqual(row["eval_status"], "human_review")
        self.assertEqual(row["failure_type"], "malformed_response")

    def test_valid_json_contract_errors_are_contract_violations(self):
        case = CASES[0]
        responses = [
            {"answer": case["answer"], "claims": [{"claim": case["answer"]}]},
            {"answer": case["answer"], "claims": [{
                "claim": case["answer"], "evidence_ids": ["e1"], "grounding": "invalid",
            }]},
            {"answer": case["answer"], "claims": [{
                "claim": case["answer"], "evidence_ids": ["missing"], "grounding": "supported",
            }]},
        ]

        class ContractErrorClient:
            def __init__(self, response):
                self.response = response

            def create(self, payload):
                return {"output_text": json.dumps(self.response)}

        for response in responses:
            row = score_case(
                case,
                run_case(case, client=ContractErrorClient(response), model="fixture"),
                1,
            )
            self.assertEqual(row["failure_type"], "contract_violation")

    def test_fixture_client_does_not_need_prompt_oracle(self):
        case = CASES[0]
        client = FixtureClient(case)
        payload = _prompt(case)
        output = json.loads(client.create(payload)["output_text"])
        self.assertNotIn("expected_claims", payload["input"])
        self.assertEqual(output["claims"][0]["grounding"], "supported")


if __name__ == "__main__":
    unittest.main()
