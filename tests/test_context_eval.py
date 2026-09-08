from __future__ import annotations

import unittest

from eval_context import CONTEXT_CASES, run_context_eval, score_context_case


class ContextEvalTests(unittest.TestCase):
    def test_cases_are_paired_and_not_tool_centered(self):
        self.assertEqual(len(CONTEXT_CASES), 13)
        self.assertEqual({case["group"] for case in CONTEXT_CASES}, {
            "context_sensitivity",
            "redundant_action",
            "stale_context_resistance",
            "conflict_resolution",
            "distractor_robustness",
            "context_sufficiency",
        })
        self.assertEqual({case["pair_id"] for case in CONTEXT_CASES}, {
            "momentum_evidence_state",
            "high52_result_availability",
            "current_truth_over_stale_state",
            "current_instruction_over_preference",
            "distractor_robustness",
            "comparison_context_sufficiency",
        })
        for case in CONTEXT_CASES:
            self.assertNotIn("expected_tool", case)
            for key in ("user_request", "context", "expected_behavior", "critical_context", "failure_if", "why"):
                self.assertIn(key, case)

    def test_rule_scorer_marks_clear_pass_and_failure(self):
        case = CONTEXT_CASES[0]
        passed = score_context_case(
            case,
            {
                "parsed": {
                    "decision": "advance_to_backtest",
                    "actions": ["propose_strategy_backtest"],
                    "used_context": case["critical_context"],
                    "reason": "strong evidence",
                },
                "parse_error": None,
                "response_text": "{}",
            },
            repeat=1,
        )
        self.assertTrue(passed["automated_pass"])
        self.assertFalse(passed["human_review_required"])

        failed = score_context_case(
            case,
            {
                "parsed": {
                    "decision": "evaluate_factor_first",
                    "actions": ["rerun_factor_evaluation"],
                    "used_context": [],
                    "reason": "not sure",
                },
                "parse_error": None,
                "response_text": "{}",
            },
            repeat=1,
        )
        self.assertFalse(failed["automated_pass"])
        self.assertTrue(failed["human_review_required"])
        self.assertTrue(failed["missing_context"])
        self.assertTrue(failed["forbidden_actions"])

    def test_fixture_runner_has_no_total_score_and_covers_all_cases(self):
        rows, meta = run_context_eval("fixture", repeats=1)
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(len(rows), len(CONTEXT_CASES))
        self.assertTrue(all(row["automated_pass"] for row in rows))
        self.assertNotIn("score", meta)
        self.assertNotIn("accuracy", meta)


if __name__ == "__main__":
    unittest.main()
