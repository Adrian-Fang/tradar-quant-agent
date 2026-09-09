from __future__ import annotations

import json
import unittest

from agent.context.eval import (
    CASES,
    _pair_results,
    _prompt,
    run_case,
    run_eval,
    score_case,
)


def _case(case_id):
    return next(case for case in CASES if case["id"] == case_id)


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
        self.assertEqual(len(CASES), 13)
        for case in CASES:
            self.assertNotIn("expected_tool", case)
            self.assertNotIn("expected_behavior", case)
            for key in ("user_request", "context", "pair_id", "group", "failure_if", "why", "behavior_contract"):
                self.assertIn(key, case)
            contract = case["behavior_contract"]
            self.assertEqual(set(contract), {"state", "decision", "action"})
            self.assertEqual(set(contract["state"]), {"must_recognize", "must_not_recognize"})
            self.assertEqual(set(contract["decision"]), {"acceptable", "must_not"})
            self.assertEqual(set(contract["action"]), {"must", "must_not", "optional"})

        self.assertEqual(
            _case("momentum_clean_context")["behavior_contract"]["action"]["must"],
            ["propose_strategy_backtest"],
        )
        self.assertEqual(
            _case("high52_fresh_result")["behavior_contract"]["action"]["must"],
            [],
        )
        self.assertEqual(
            _case("strategy_compare_result_missing")["behavior_contract"]["action"]["must"],
            ["block_for_missing_result"],
        )

    def test_prompt_uses_pair_tags_without_case_oracle(self):
        case = _case("momentum_strong_evidence")
        body = json.loads(_prompt(case)["input"])
        pair_cases = [candidate for candidate in CASES if candidate["pair_id"] == case["pair_id"]]
        expected_decisions = {
            tag
            for candidate in pair_cases
            for key in ("acceptable", "must_not")
            for tag in candidate["behavior_contract"]["decision"][key]
        }
        expected_state_tags = {
            tag
            for candidate in pair_cases
            for key in ("must_recognize", "must_not_recognize")
            for tag in candidate["behavior_contract"]["state"][key]
        }
        expected_action_tags = {
            tag
            for candidate in pair_cases
            for key in ("must", "must_not", "optional")
            for tag in candidate["behavior_contract"]["action"][key]
        }
        self.assertEqual(body["user_request"], case["user_request"])
        self.assertEqual(body["context"], case["context"])
        self.assertEqual(
            set(body),
            {"user_request", "context", "allowed_state_tags", "allowed_decisions", "allowed_action_tags"},
        )
        self.assertEqual(set(body["allowed_state_tags"]), expected_state_tags)
        self.assertEqual(set(body["allowed_decisions"]), expected_decisions)
        self.assertEqual(set(body["allowed_action_tags"]), expected_action_tags)
        self.assertNotIn("reuse_existing_result", body["allowed_decisions"])
        for leaked_key in ("critical_context", "expected_behavior", "behavior_contract", "must_recognize"):
            self.assertNotIn(leaked_key, body)

    def test_fixture_passes_false_negative_cases(self):
        rows, meta = run_eval("fixture", repeats=1)
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(row["status"] == "pass" for row in rows))
        by_id = {row["case"]: row for row in rows}
        for case_id in (
            "momentum_clean_context",
            "momentum_with_distractors",
            "high52_fresh_result",
            "high52_no_result",
            "current_no_extra_validation",
        ):
            self.assertEqual(by_id[case_id]["status"], "pass")
        self.assertNotIn("score", meta)
        self.assertNotIn("accuracy", meta)

    def test_missing_state_is_partial_but_not_case_failure(self):
        case = _case("momentum_strong_evidence")
        row = score_case(
            case,
            _outcome(case, state=case["behavior_contract"]["state"]["must_recognize"][:1]),
            repeat=1,
        )
        self.assertEqual(row["state_status"], "partial")
        self.assertFalse(row["state_interpretation_pass"])
        self.assertEqual(row["status"], "pass")

    def test_state_contradiction_is_deterministic_failure(self):
        case = _case("momentum_strong_evidence")
        state = case["behavior_contract"]["state"]["must_recognize"] + [
            case["behavior_contract"]["state"]["must_not_recognize"][0]
        ]
        row = score_case(case, _outcome(case, state=state), repeat=1)
        self.assertEqual(row["state_status"], "fail")
        self.assertEqual(row["status"], "fail")
        self.assertFalse(row["human_review_required"])

    def test_decision_and_action_contract_violations_fail(self):
        case = _case("momentum_clean_context")
        bad_decision = score_case(case, _outcome(case, decision="stop_and_reassess"), repeat=1)
        self.assertEqual(bad_decision["status"], "fail")
        bad_action = score_case(case, _outcome(case, actions=[]), repeat=1)
        self.assertEqual(bad_action["status"], "fail")
        empty_decision = score_case(case, _outcome(case, decision=""), repeat=1)
        self.assertEqual(empty_decision["status"], "fail")

    def test_malformed_and_provider_errors_require_human_review(self):
        case = CASES[0]

        class MalformedClient:
            def create(self, payload):
                return {"output_text": "{}"}

        malformed = run_case(case, client=MalformedClient(), model="fixture")
        malformed_row = score_case(case, malformed, repeat=1)
        self.assertEqual(malformed_row["status"], "human_review")
        self.assertTrue(malformed_row["human_review_required"])

        class ProviderErrorClient:
            def create(self, payload):
                raise RuntimeError("provider unavailable")

        provider_error = run_case(case, client=ProviderErrorClient(), model="fixture")
        provider_row = score_case(case, provider_error, repeat=1)
        self.assertEqual(provider_row["status"], "human_review")
        self.assertEqual(provider_row["failure_type"], "provider_error")

    def test_acceptable_alternatives_and_nonblocking_missing_preference(self):
        momentum = _case("momentum_not_started")
        momentum_row = score_case(
            momentum,
            _outcome(momentum, decision="blocked_pending_factor_evaluation", actions=[]),
            repeat=1,
        )
        self.assertEqual(momentum_row["status"], "pass")

        high52 = _case("high52_no_result")
        high52_row = score_case(
            high52,
            _outcome(high52, decision="blocked_missing_result", actions=["start_high52_evaluation"]),
            repeat=1,
        )
        self.assertEqual(high52_row["status"], "pass")

        no_validation = _case("current_no_extra_validation")
        no_validation_row = score_case(
            no_validation,
            _outcome(no_validation, decision="only_restate_old_result", actions=[]),
            repeat=1,
        )
        self.assertEqual(no_validation_row["status"], "pass")

    def test_unspecified_metric_block_is_a_contract_failure(self):
        case = _case("strategy_compare_metrics_unspecified")
        row = score_case(
            case,
            _outcome(case, decision="blocked_for_unspecified_metric", actions=[]),
            repeat=1,
        )
        self.assertEqual(row["status"], "fail")
        self.assertFalse(row["human_review_required"])
        self.assertFalse(row["decision_policy_pass"])

    def test_pair_results_are_independent_from_case_failures(self):
        clean = _case("momentum_clean_context")
        distractors = _case("momentum_with_distractors")
        broken_state = clean["behavior_contract"]["state"]["must_recognize"] + [
            clean["behavior_contract"]["state"]["must_not_recognize"][0]
        ]
        broken = score_case(
            clean,
            _outcome(clean, state=broken_state, actions=["propose_strategy_backtest"]),
            repeat=1,
        )
        valid = score_case(
            distractors,
            _outcome(distractors, actions=["propose_strategy_backtest"]),
            repeat=1,
        )
        self.assertEqual(broken["status"], "fail")
        pair = next(result for result in _pair_results([broken, valid]) if result["pair_id"] == "distractor_robustness")
        self.assertEqual(pair["repeat_results"][0]["status"], "pass")

    def test_pair_results_are_per_repeat_and_ignore_optional_actions(self):
        rows, meta = run_eval("fixture", repeats=3)
        pair_results = {result["pair_id"]: result for result in meta["pair_results"]}
        momentum = pair_results["momentum_evidence_state"]
        self.assertEqual(len(momentum["repeat_results"]), 3)
        self.assertTrue(all(repeat["pair_pass"] for repeat in momentum["repeat_results"]))

        clean = _case("momentum_clean_context")
        distractors = _case("momentum_with_distractors")
        with_optional = score_case(
            clean,
            _outcome(clean, actions=["propose_strategy_backtest", "use_momentum_result"]),
            repeat=1,
        )
        without_optional = score_case(
            distractors,
            _outcome(distractors, actions=["propose_strategy_backtest"]),
            repeat=1,
        )
        pair = next(
            result
            for result in _pair_results([with_optional, without_optional])
            if result["pair_id"] == "distractor_robustness"
        )
        repeat = pair["repeat_results"][0]
        self.assertEqual(repeat["distinct_signature_count"], 1)
        self.assertTrue(repeat["pair_pass"])


if __name__ == "__main__":
    unittest.main()
