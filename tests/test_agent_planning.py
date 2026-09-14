from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.planning.eval import CASES, FixtureClient, _prompt, run_case, run_eval, score_case


class PlanningEvalTests(unittest.TestCase):
    def test_dataset_has_planning_contract_fields(self):
        self.assertEqual(len(CASES), 12)
        for case in CASES:
            self.assertEqual(
                set(case), {"id", "request", "slice", "expected_status", "expected_steps"}
            )
            self.assertIn(case["expected_status"], {"ready", "needs_input", "no_action"})
            self.assertIsInstance(case["expected_steps"], list)
            for step in case["expected_steps"]:
                self.assertEqual(set(step), {"name", "arguments"})

    def test_prompt_contains_request_and_schemas_without_oracle(self):
        case = CASES[0]
        payload = _prompt(case)
        prompt_input = json.loads(payload["input"])

        self.assertEqual(set(prompt_input), {"user_request", "tool_schemas"})
        self.assertEqual(prompt_input["user_request"], case["request"])
        self.assertNotIn("expected_status", payload["input"])
        self.assertNotIn("expected_steps", payload["input"])
        self.assertNotIn(case["slice"], payload["input"])

    def test_fixture_ready_single_step_passes(self):
        case = CASES[1]
        row = score_case(case, run_case(case, client=FixtureClient(case), model=""), 1)

        self.assertTrue(row["case_pass"])
        self.assertEqual(row["required_step_recall"], 1.0)
        self.assertEqual(row["argument_accuracy"], 1.0)

    def test_extra_step_is_penalized(self):
        case = CASES[1]
        outcome = {
            "parsed": {
                "status": "ready",
                "steps": case["expected_steps"] + [{"name": "inspect_universe", "arguments": {}}],
                "reason": "",
            },
            "parse_error": "",
        }

        row = score_case(case, outcome, 1)

        self.assertFalse(row["case_pass"])
        self.assertLess(row["step_precision"], 1.0)
        self.assertGreater(row["unnecessary_step_rate"], 0.0)

    def test_missing_required_step_and_wrong_args_fail(self):
        case = CASES[4]
        missing = {
            "parsed": {
                "status": "ready",
                "steps": [case["expected_steps"][0]],
                "reason": "",
            },
            "parse_error": "",
        }
        wrong_args = {
            "parsed": {
                "status": "ready",
                "steps": [{
                    "name": "evaluate_factor",
                    "arguments": {"factor": "wrong_factor", "observe_start": "2025-09-01", "observe_end": "2026-08-31"},
                }, case["expected_steps"][1]],
                "reason": "",
            },
            "parse_error": "",
        }

        missing_row = score_case(case, missing, 1)
        wrong_args_row = score_case(case, wrong_args, 1)
        self.assertLess(missing_row["required_step_recall"], 1.0)
        self.assertFalse(missing_row["case_pass"])
        self.assertLess(wrong_args_row["argument_accuracy"], 1.0)
        self.assertFalse(wrong_args_row["case_pass"])

    def test_needs_input_and_no_action_require_empty_plan(self):
        for case in CASES[7:]:
            if case["expected_status"] not in {"needs_input", "no_action"}:
                continue
            passing = score_case(case, run_case(case, client=FixtureClient(case), model=""), 1)
            self.assertTrue(passing["stop_correct"])
            self.assertTrue(passing["case_pass"])

            violating = {
                "parsed": {
                    "status": case["expected_status"],
                    "steps": [{"name": "inspect_universe", "arguments": {}}],
                    "reason": "guessed",
                },
                "parse_error": "",
            }
            row = score_case(case, violating, 1)
            self.assertFalse(row["case_pass"])

    def test_malformed_response_is_eval_failure(self):
        class MalformedClient:
            def create(self, payload):
                return {"output_text": "not json"}

        row = score_case(CASES[0], run_case(CASES[0], client=MalformedClient(), model=""), 1)

        self.assertEqual(row["eval_status"], "human_review")
        self.assertEqual(row["failure_type"], "malformed_response")
        self.assertFalse(row["case_pass"])

    def test_fixture_full_pass(self):
        rows, meta = run_eval(provider="fixture", repeats=1)

        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["case_pass"] for row in rows))
        self.assertEqual(meta["metrics"]["case_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
