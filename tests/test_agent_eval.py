from __future__ import annotations

import copy
import unittest

from agent.agent_eval import CASES, _metrics, run_eval, score_case


class WholeSystemEvalTests(unittest.TestCase):
    def test_dataset_schema_keeps_oracle_out_of_observed_trace(self):
        self.assertEqual(len(CASES), 14)
        for case in CASES:
            self.assertIn("id", case)
            self.assertIn("request", case)
            self.assertIn("expected", case)
            self.assertIn("observed", case)
            self.assertIn("required_steps", case["expected"])
            self.assertIn("planning", case["observed"])
            self.assertIn("orchestration", case["observed"])
            self.assertNotIn("expected", case["observed"])

    def test_fixture_all_cases_pass(self):
        rows, meta = run_eval()
        self.assertEqual(len(rows), 14)
        self.assertTrue(all(row["case_pass"] for row in rows))
        self.assertEqual(meta["metrics"]["case_pass_rate"], 1.0)
        self.assertEqual(meta["metrics"]["outcome_pass_rate"], 13 / 14)
        self.assertEqual(meta["metrics"]["failure_attribution_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["eval_failures"], 0)

    def test_trajectory_metrics_expose_missing_extra_and_order(self):
        missing = next(case for case in CASES if case["id"] == "missing_required_planning_step")
        missing_row = score_case(missing, missing["observed"])
        self.assertEqual(missing_row["trajectory"]["required_step_recall"], 0.5)

        extra = next(case for case in CASES if case["id"] == "extra_unnecessary_step")
        extra_row = score_case(extra, extra["observed"])
        self.assertEqual(extra_row["trajectory"]["precision"], 0.5)
        self.assertEqual(extra_row["trajectory"]["unnecessary_step_rate"], 0.5)

        ordered = next(case for case in CASES if case["id"] == "multi_step_research_success")
        reversed_observed = copy.deepcopy(ordered["observed"])
        reversed_observed["steps"].reverse()
        reversed_row = score_case(ordered, reversed_observed)
        self.assertEqual(reversed_row["trajectory"]["order_correctness"], 0.0)
        self.assertEqual(reversed_row["failure_stage"], "orchestration")

    def test_failure_attribution_covers_component_stages(self):
        expected_stages = {
            "context_missing_required_input": "context",
            "missing_required_planning_step": "planning",
            "extra_unnecessary_step": "planning",
            "wrong_tool_arguments": "tool",
            "tool_execution_failure": "execution",
            "grounding_unsupported_and_contradicted_claims": "grounding",
            "state_lifecycle_mismatch": "state",
            "orchestration_level_failure": "orchestration",
            "wrong_final_outcome_after_valid_trace": "outcome",
        }
        for case_id, stage in expected_stages.items():
            case = next(case for case in CASES if case["id"] == case_id)
            row = score_case(case, case["observed"])
            self.assertEqual(row["failure_stage"], stage)
            self.assertTrue(row["case_pass"])

        for case_id in ("retrieval_abstention", "hitl_blocked_product_boundary"):
            case = next(case for case in CASES if case["id"] == case_id)
            row = score_case(case, case["observed"])
            self.assertIsNone(row["failure_stage"])
            self.assertTrue(row["case_pass"])

    def test_malformed_observed_envelope_is_eval_failure(self):
        case = CASES[0]
        malformed = copy.deepcopy(case["observed"])
        del malformed["steps"]
        row = score_case(case, malformed)
        self.assertEqual(row["eval_status"], "human_review")
        self.assertEqual(row["eval_failure_type"], "contract_violation")
        self.assertFalse(row["case_pass"])

    def test_eval_failure_counts_in_metrics_denominator(self):
        case = CASES[0]
        valid = score_case(case, case["observed"])
        invalid = score_case(case, {"steps": []})
        metrics = _metrics([valid, invalid])
        self.assertEqual(metrics["case_pass_rate"], 0.5)
        self.assertEqual(metrics["eval_failures"], 1)
        self.assertEqual(metrics["failure_breakdown"], {"contract_violation": 1})

    def test_failure_field_is_debug_only(self):
        case = next(case for case in CASES if case["id"] == "wrong_final_outcome_after_valid_trace")
        without_debug = copy.deepcopy(case["observed"])
        without_debug.pop("failure")
        wrong_debug = copy.deepcopy(without_debug)
        wrong_debug["failure"] = {"stage": "context", "type": "made_up"}
        other_mismatch = copy.deepcopy(without_debug)
        other_mismatch["grounding"] = {"fully_grounded": False, "labels": ["unverifiable"]}

        self.assertTrue(score_case(case, without_debug)["case_pass"])
        self.assertTrue(score_case(case, wrong_debug)["case_pass"])
        self.assertFalse(score_case(case, other_mismatch)["case_pass"])

    def test_ambiguous_execution_trace_is_unclassified_orchestration(self):
        case = CASES[0]
        observed = copy.deepcopy(case["observed"])
        observed["steps"] = []
        row = score_case(case, observed)
        self.assertEqual(row["failure_stage"], "orchestration")
        self.assertEqual(row["failure_type"], "trajectory_mismatch")

    def test_planning_stops_allow_unreached_stages(self):
        for status in ("needs_input", "no_action"):
            case = {
                "id": f"planning_{status}",
                "expected": {
                    "outcome": status,
                    "context_ids": ["request_scope"],
                    "retrieval_status": "not_used",
                    "retrieval_ids": [],
                    "required_steps": [],
                    "planning_status": status,
                    "run_status": None,
                    "final_status": None,
                    "grounding": None,
                    "hitl_decision": None,
                    "failure_stage": None,
                    "failure_type": None,
                },
                "observed": {
                    "context": {"selected_ids": ["request_scope"]},
                    "planning": {"status": status, "steps": []},
                    "steps": [],
                    "retrieval": None,
                    "research_run": None,
                    "grounding": None,
                    "hitl": None,
                    "orchestration": None,
                    "outcome": {"status": status},
                },
            }
            self.assertTrue(score_case(case, case["observed"])["case_pass"])

    def test_repeat_rows_are_deterministic(self):
        rows, meta = run_eval(repeats=2)
        self.assertEqual(len(rows), 28)
        self.assertEqual(meta["repeats"], 2)
        self.assertEqual(
            [(row["case"], row["repeat"], row["case_pass"]) for row in rows[:2]],
            [("simple_read_only_success", 1, True), ("simple_read_only_success", 2, True)],
        )


if __name__ == "__main__":
    unittest.main()
