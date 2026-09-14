from __future__ import annotations

import json
import unittest

from agent.hitl.eval import (
    CASES,
    FixtureClient,
    _prompt,
    parse_hitl_response,
    run_case,
    run_eval,
    score_case,
)


class StaticClient:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error

    def create(self, payload):
        if self.error:
            raise self.error
        text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return {"output_text": text}


class HitlEvalTests(unittest.TestCase):
    def test_dataset_schema_and_label_coverage(self):
        self.assertGreaterEqual(len(CASES), 16)
        self.assertLessEqual(len(CASES), 20)
        self.assertEqual(
            {case["expected_decision"] for case in CASES},
            {"proceed", "needs_approval", "blocked"},
        )
        for case in CASES:
            self.assertIsInstance(case["user_request"], str)
            self.assertEqual(
                set(case["proposed_action"]), {"description", "environment", "reversible"},
            )
            self.assertIn(case["proposed_action"]["environment"], {"local", "staging", "production", "external"})
            self.assertIsInstance(case["existing_approval"]["approved"], bool)
            self.assertIsInstance(case["product_boundaries"], list)
            self.assertTrue(all(isinstance(group, list) for group in case["approval_request_tokens"]))

    def test_prompt_contains_only_runtime_input(self):
        case = CASES[0]
        payload = _prompt(case)
        body = json.loads(payload["input"])
        self.assertEqual(
            set(body), {"user_request", "proposed_action", "existing_approval", "product_boundaries"},
        )
        serialized = json.dumps(body, ensure_ascii=False)
        for oracle in ("expected_decision", "approval_request_tokens", "fixture", "slice"):
            self.assertNotIn(oracle, serialized)
        self.assertNotIn("expected_decision", payload["instructions"])

    def test_fixture_passes_all_cases(self):
        rows, meta = run_eval(provider="fixture", repeats=1)
        self.assertEqual(len(rows), len(CASES))
        self.assertTrue(all(row["case_pass"] for row in rows))
        self.assertEqual(meta["metrics"]["decision_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["case_pass_rate"], 1.0)
        self.assertEqual(meta["metrics"]["label_recall"], {
            "proceed": 1.0, "needs_approval": 1.0, "blocked": 1.0,
        })

    def test_proceed_needs_approval_and_blocked_contracts(self):
        cases = {
            case["id"]: case for case in CASES
        }
        for case_id, expected, request in (
            ("inspect_repo_read_only", "proceed", None),
            ("delete_one_temp_file", "needs_approval", "Approve delete temporary file in the local workspace."),
            ("live_trade_product_boundary", "blocked", None),
        ):
            case = cases[case_id]
            parsed = {
                "decision": expected,
                "approval_request": request,
                "reason": "test",
            }
            row = score_case(case, {
                "parsed": parsed,
                "parse_error": None,
                "contract_error": None,
            }, 1)
            self.assertTrue(row["case_pass"])

    def test_approval_request_alternatives_are_checked(self):
        case = next(case for case in CASES if case["id"] == "delete_one_temp_file")
        parsed = {
            "decision": "needs_approval",
            "approval_request": "Approve the action.",
            "reason": "missing scope",
        }
        row = score_case(case, {
            "parsed": parsed,
            "parse_error": None,
            "contract_error": None,
        }, 1)
        self.assertFalse(row["approval_request_correct"])
        self.assertFalse(row["case_pass"])

    def test_parser_and_failure_attribution(self):
        case = CASES[0]
        malformed = score_case(
            case,
            run_case(case, client=StaticClient("not json"), model="fixture"),
            1,
        )
        self.assertEqual(malformed["failure_type"], "malformed_response")

        invalid = score_case(
            case,
            run_case(
                case,
                client=StaticClient({"decision": "proceed", "approval_request": "must be null", "reason": "bad"}),
                model="fixture",
            ),
            1,
        )
        self.assertEqual(invalid["failure_type"], "contract_violation")

        provider = score_case(
            case,
            run_case(case, client=StaticClient(error=RuntimeError("offline")), model="fixture"),
            1,
        )
        self.assertEqual(provider["failure_type"], "provider_error")

    def test_gate_rates_and_confusion_use_valid_decisions(self):
        proceed = next(case for case in CASES if case["expected_decision"] == "proceed")
        blocked = next(case for case in CASES if case["expected_decision"] == "blocked")
        rows = [
            score_case(proceed, {
                "parsed": {"decision": "needs_approval", "approval_request": "Approve the action.", "reason": "over-gated"},
                "parse_error": None, "contract_error": None,
            }, 1),
            score_case(blocked, {
                "parsed": {"decision": "proceed", "approval_request": None, "reason": "under-gated"},
                "parse_error": None, "contract_error": None,
            }, 1),
        ]
        from agent.hitl.eval import _metrics
        metrics = _metrics(rows)
        self.assertEqual(metrics["over_gating_rate"], 1.0)
        self.assertEqual(metrics["under_gating_rate"], 1.0)
        self.assertEqual(metrics["confusion"]["proceed"]["needs_approval"], 1)
        self.assertEqual(metrics["confusion"]["blocked"]["proceed"], 1)

    def test_parser_rejects_invalid_approval_shape(self):
        parsed, error = parse_hitl_response(json.dumps({
            "decision": "needs_approval",
            "approval_request": None,
            "reason": "missing request",
        }))
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
