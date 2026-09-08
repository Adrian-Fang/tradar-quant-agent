from __future__ import annotations

import json
import unittest

from agent.context.eval import (
    CONTEXT_CASES,
    _prompt,
    run_context_case,
    run_context_eval,
    score_context_case,
)


def _outcome(case, *, state=None, decision=None, actions=None, reason="test"):
    contract = case["behavior_contract"]
    parsed = {
        "state": state if state is not None else contract["state"]["must_recognize"],
        "decision": decision if decision is not None else contract["decision"]["acceptable"][0],
        "actions": actions if actions is not None else contract["action"]["must"],
        "reason": reason,
    }
    return {
        "parsed": parsed,
        "parse_error": None,
        "structure_error": None,
        "response_text": json.dumps(parsed),
    }


class ContextEvalTests(unittest.TestCase):
    def test_cases_use_behavior_contract_schema(self):
        self.assertEqual(len(CONTEXT_CASES), 13)
        self.assertEqual({case["group"] for case in CONTEXT_CASES}, {
            "context_sensitivity",
            "redundant_action",
            "stale_context_resistance",
            "conflict_resolution",
            "distractor_robustness",
            "context_sufficiency",
        })
        for case in CONTEXT_CASES:
            self.assertNotIn("expected_tool", case)
            for key in ("user_request", "context", "pair_id", "group", "failure_if", "why", "behavior_contract"):
                self.assertIn(key, case)
            contract = case["behavior_contract"]
            self.assertEqual(set(contract), {"state", "decision", "action"})
            self.assertEqual(set(contract["state"]), {"must_recognize", "must_not_recognize"})
            self.assertEqual(set(contract["decision"]), {"acceptable", "must_not"})
            self.assertEqual(set(contract["action"]), {"must", "must_not", "optional"})

    def test_prompt_has_global_tags_but_no_case_oracle(self):
        case = CONTEXT_CASES[0]
        payload = _prompt(case)
        body = json.loads(payload["input"])
        self.assertEqual(body["user_request"], case["user_request"])
        self.assertEqual(body["context"], case["context"])
        for leaked_key in ("critical_context", "expected_behavior", "behavior_contract", "must_recognize"):
            self.assertNotIn(leaked_key, body)
        self.assertTrue(set(case["behavior_contract"]["state"]["must_recognize"]) <= set(body["allowed_state_tags"]))
        self.assertTrue(body["allowed_decisions"])
        self.assertTrue(body["allowed_action_tags"])

    def test_fixture_runner_passes_all_cases_without_aggregate_score(self):
        rows, meta = run_context_eval("fixture", repeats=1)
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(row["status"] == "pass" for row in rows))
        self.assertNotIn("score", meta)
        self.assertNotIn("accuracy", meta)
        self.assertEqual(meta["pair_results"][-1]["pair_id"], "comparison_context_sufficiency")

    def test_valid_contract_violation_is_fail_not_human_review(self):
        case = CONTEXT_CASES[0]
        row = score_context_case(
            case,
            _outcome(case, decision="stop_and_reassess"),
            repeat=1,
        )
        self.assertEqual(row["status"], "fail")
        self.assertFalse(row["human_review_required"])
        self.assertFalse(row["decision_policy_pass"])
        self.assertEqual(row["failure_type"], "behavior_contract_violation")

    def test_malformed_and_provider_errors_require_human_review(self):
        case = CONTEXT_CASES[0]

        class MalformedClient:
            def create(self, payload):
                return {"output_text": "{}"}

        malformed = run_context_case(case, client=MalformedClient(), model="fixture")
        malformed_row = score_context_case(case, malformed, repeat=1)
        self.assertEqual(malformed_row["status"], "human_review")
        self.assertTrue(malformed_row["human_review_required"])

        class ProviderErrorClient:
            def create(self, payload):
                raise RuntimeError("provider unavailable")

        provider_error = run_context_case(case, client=ProviderErrorClient(), model="fixture")
        provider_row = score_context_case(case, provider_error, repeat=1)
        self.assertEqual(provider_row["status"], "human_review")
        self.assertEqual(provider_row["failure_type"], "provider_error")

    def test_momentum_not_started_accepts_blocked_pending_decision(self):
        case = next(case for case in CONTEXT_CASES if case["id"] == "momentum_not_started")
        row = score_context_case(
            case,
            _outcome(case, decision="blocked_pending_factor_evaluation", actions=[]),
            repeat=1,
        )
        self.assertEqual(row["status"], "pass")
        self.assertTrue(row["decision_policy_pass"])
        self.assertTrue(row["action_compliance_pass"])

    def test_unspecified_metric_block_is_a_contract_failure(self):
        case = next(case for case in CONTEXT_CASES if case["id"] == "strategy_compare_metrics_unspecified")
        row = score_context_case(
            case,
            _outcome(case, decision="blocked_for_unspecified_metric", actions=[]),
            repeat=1,
        )
        self.assertEqual(row["status"], "fail")
        self.assertFalse(row["human_review_required"])
        self.assertFalse(row["decision_policy_pass"])

    def test_pair_change_and_no_change_contracts(self):
        rows, meta = run_context_eval("fixture", repeats=1)
        pair_results = {result["pair_id"]: result for result in meta["pair_results"]}
        self.assertEqual(pair_results["distractor_robustness"]["status"], "pass")
        self.assertTrue(pair_results["distractor_robustness"]["pair_pass"])
        self.assertEqual(pair_results["momentum_evidence_state"]["status"], "pass")
        self.assertTrue(pair_results["momentum_evidence_state"]["pair_pass"])
        self.assertGreater(pair_results["momentum_evidence_state"]["distinct_signature_count"], 1)
        self.assertEqual(pair_results["current_truth_over_stale_state"]["status"], "not_scored")


if __name__ == "__main__":
    unittest.main()
