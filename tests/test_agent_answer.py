from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from agent.answer.eval import CASES, FixtureClient, _prompt, run_case, run_eval, score_case
from agent.answer.synthesizer import parse_synthesis_response, synthesize_answer


EVIDENCE = [{"id": "e1", "text": "The result is positive."}]


class FakeClient:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return {"output_text": self.output if isinstance(self.output, str) else json.dumps(self.output)}


def response(status="success", answer="The result is positive.", evidence_ids=None):
    return {
        "status": status,
        "answer": answer,
        "evidence_ids": ["e1"] if evidence_ids is None else evidence_ids,
    }


class AnswerSynthesisTests(unittest.TestCase):
    def test_dataset_schema_and_prompt_has_no_oracle(self):
        self.assertGreaterEqual(len(CASES), 6)
        for case in CASES:
            self.assertIn("id", case)
            self.assertIn("user_request", case)
            self.assertIn("evidence", case)
            self.assertIn("expected", case)
            self.assertTrue(case["evidence"])
            self.assertEqual(
                set(_prompt(case)["input"] and json.loads(_prompt(case)["input"])),
                {"user_request", "evidence"},
            )
            self.assertNotIn("required_content", _prompt(case)["input"])
            self.assertNotIn("fixture_answer", _prompt(case)["input"])

    def test_parser_rejects_malformed_unknown_and_invalid_shape(self):
        parsed, error = parse_synthesis_response("not json", {"e1"})
        self.assertIsNone(parsed)
        self.assertIsNotNone(error)

        parsed, error = parse_synthesis_response(
            json.dumps(response(evidence_ids=["missing"])), {"e1"},
        )
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(error)

        parsed, error = parse_synthesis_response(
            json.dumps({**response(), "reason": "extra"}), {"e1"},
        )
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(error)

        parsed, error = parse_synthesis_response(
            json.dumps({"status": "success"}), {"e1"},
        )
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(error)

    def test_runtime_success_and_insufficient_evidence(self):
        for status in ("success", "insufficient_evidence"):
            result = synthesize_answer(
                "What happened?",
                EVIDENCE,
                client=FakeClient(output=response(status=status)),
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["result"]["status"], status)

    def test_runtime_provider_and_malformed_fail_closed(self):
        provider = synthesize_answer(
            "What happened?", EVIDENCE, client=FakeClient(error=RuntimeError("offline")),
        )
        self.assertEqual((provider["status"], provider["error_type"]), ("error", "provider_error"))
        malformed = synthesize_answer(
            "What happened?", EVIDENCE, client=FakeClient(output="not json"),
        )
        self.assertEqual((malformed["status"], malformed["error_type"]), ("error", "malformed_response"))

    def test_runtime_prompt_uses_only_request_and_evidence(self):
        client = FakeClient(output=response())
        synthesize_answer("What happened?", EVIDENCE, client=client)
        body = json.loads(client.calls[0]["input"])
        self.assertEqual(set(body), {"user_request", "evidence"})

    def test_prompt_rejects_unsupported_qualitative_and_recommendation_overreach(self):
        prompt = _prompt(CASES[0])["instructions"]
        self.assertIn("Separate directly observed facts from interpretation", prompt)
        self.assertIn("sufficient, large, small, strong, or weak", prompt)
        self.assertIn("should use, avoid, or use", prompt)
        self.assertIn("does not establish causality", prompt)

    def test_evidence_ids_are_minimal_but_keep_required_multi_evidence(self):
        insufficient = next(
            case for case in CASES if case["id"] == "insufficient_intraday_vwap_claim"
        )
        answer = "Insufficient evidence: e1 shows the strategy was not tested, so the requested benefit cannot be established."
        row = score_case(insufficient, run_case(
            insufficient,
            client=FakeClient(output=response(
                status="insufficient_evidence",
                answer=answer,
                evidence_ids=["e1"],
            )),
            model="fixture",
        ), 1)
        self.assertTrue(row["case_pass"])
        self.assertEqual(row["actual_evidence_ids"], ["e1"])

        multi = next(case for case in CASES if case["id"] == "relative_reversal_synthesis")
        row = score_case(multi, run_case(
            multi,
            client=FakeClient(output=response(
                answer=multi["expected"]["fixture_answer"],
                evidence_ids=["e1", "e2"],
            )),
            model="fixture",
        ), 1)
        self.assertTrue(row["case_pass"])
        self.assertEqual(row["actual_evidence_ids"], ["e1", "e2"])

    def test_fixture_scores_all_cases(self):
        rows, meta = run_eval(provider="fixture", repeats=1)
        self.assertEqual(len(rows), len(CASES))
        self.assertTrue(all(row["case_pass"] for row in rows))
        self.assertEqual(meta["metrics"]["status_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["evidence_selection_precision"], 1.0)
        self.assertEqual(meta["metrics"]["evidence_selection_recall"], 1.0)
        self.assertEqual(meta["metrics"]["required_content_coverage"], 1.0)
        self.assertEqual(meta["metrics"]["forbidden_content_violation_rate"], 0.0)
        self.assertEqual(meta["metrics"]["case_pass_rate"], 1.0)

    def test_content_and_evidence_mismatch_fails(self):
        case = CASES[0]
        outcome = run_case(case, client=FixtureClient(case), model="fixture")
        actual = json.loads(outcome["response_text"])
        actual["answer"] = "An unsupported invented answer."
        row = score_case(case, {
            **outcome,
            "response_text": json.dumps(actual),
            "parsed": actual,
        }, 1)
        self.assertFalse(row["case_pass"])
        self.assertEqual(row["failure_type"], "synthesis_mismatch")

    def test_negated_uncertain_answer_is_not_false_positive(self):
        case = next(case for case in CASES if case["id"] == "insufficient_future_extrapolation")
        parsed = response(
            status="insufficient_evidence",
            answer="The evidence does not prove the effect will remain profitable; the record ends on 2026-07-31.",
            evidence_ids=["e1"],
        )
        row = score_case(case, {
            "response_text": json.dumps(parsed),
            "parsed": parsed,
            "parse_error": None,
            "contract_error": None,
        }, 1)
        self.assertTrue(row["case_pass"])

    def test_clear_numeric_or_directional_claim_still_fails(self):
        case = CASES[0]
        parsed = response(
            answer="The canonical defaults are 25 bp for buys and 15 bp for sells.",
        )
        row = score_case(case, {
            "response_text": json.dumps(parsed),
            "parsed": parsed,
            "parse_error": None,
            "contract_error": None,
        }, 1)
        self.assertFalse(row["case_pass"])
        self.assertGreater(row["forbidden_content_violations"], 0)

    def test_provider_skip_behavior(self):
        for provider, key in (("deepseek", "DEEPSEEK_API_KEY"), ("openai", "OPENAI_API_KEY")):
            with self.subTest(provider=provider), patch.dict(os.environ, {key: ""}), patch(
                "agent.answer.eval.DeepSeekChatClient" if provider == "deepseek"
                else "agent.answer.eval.OpenAIResponsesClient"
            ) as client:
                rows, meta = run_eval(provider=provider, repeats=1)
            self.assertEqual(rows, [])
            self.assertEqual(meta["status"], "skipped")
            self.assertIn(key, meta["reason"])
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
