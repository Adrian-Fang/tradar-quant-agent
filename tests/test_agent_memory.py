from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.memory.eval import (
    CASES,
    FixtureClient,
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
        "contract_error": None,
    }


class MemoryEvalTests(unittest.TestCase):
    def test_dataset_schema_and_action_coverage(self):
        self.assertGreaterEqual(len(CASES), 14)
        self.assertLessEqual(len(CASES), 18)
        self.assertEqual({case["expected"]["action"] for case in CASES}, {"write", "update", "ignore"})
        for case in CASES:
            self.assertIsInstance(case["message"], str)
            self.assertIsInstance(case["existing_memories"], list)
            self.assertIn(case["expected"]["action"], {"write", "update", "ignore"})
            if case["expected"]["action"] == "update":
                self.assertIn(
                    case["expected"]["supersedes_id"],
                    {item["id"] for item in case["existing_memories"]},
                )
            for field in ("required_tokens", "forbidden_tokens"):
                self.assertTrue(all(isinstance(group, list) for group in case["expected"][field]))

    def test_prompt_contains_only_message_and_existing_memories(self):
        case = CASES[0]
        payload = _prompt(case)
        body = json.loads(payload["input"])
        self.assertEqual(set(body), {"message", "existing_memories"})
        serialized = json.dumps(body, ensure_ascii=False)
        for oracle in ("expected", "fixture_memory", "slice", "difficulty", "why"):
            self.assertNotIn(oracle, serialized)
        self.assertNotIn("required_tokens", payload["instructions"])
        self.assertIn("atomic durable proposition", payload["instructions"])
        self.assertIn("can coexist", payload["instructions"])
        self.assertIn("separate requirement", payload["instructions"])

    def test_same_topic_cases_use_the_same_memory_slot(self):
        cases = [
            case for case in CASES
            if case["id"] in {"same_topic_additional_requirement", "same_topic_replacement_requirement"}
        ]
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["existing_memories"], cases[1]["existing_memories"])
        self.assertEqual(cases[0]["existing_memories"][0]["id"], "m1")
        self.assertIn("Research summaries", cases[0]["existing_memories"][0]["text"])
        self.assertEqual(cases[0]["expected"]["action"], "write")
        self.assertIsNone(cases[0]["expected"]["supersedes_id"])
        self.assertEqual(cases[1]["expected"]["action"], "update")
        self.assertEqual(cases[1]["expected"]["supersedes_id"], "m1")

    def test_fixture_passes_all_cases(self):
        rows, meta = run_eval(provider="fixture", repeats=1)
        self.assertEqual(len(rows), len(CASES))
        self.assertTrue(all(row["case_pass"] for row in rows))
        self.assertEqual(meta["metrics"]["action_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["supersedes_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["memory_content_accuracy"], 1.0)
        self.assertEqual(meta["metrics"]["case_pass_rate"], 1.0)

    def test_eval_failures_count_in_case_pass_rate_and_failure_rate(self):
        case = CASES[0]
        valid = outcome({
            "action": "write",
            "memory": case["fixture_memory"],
            "supersedes_id": None,
            "reason": "valid",
        })
        contract_failure = {
            "response_text": "{}",
            "parsed": {},
            "parse_error": None,
            "contract_error": "invalid response",
        }
        with patch("agent.memory.eval.CASES", (case, case)), patch(
            "agent.memory.eval.run_case", side_effect=[valid, contract_failure],
        ):
            rows, meta = run_eval(provider="fixture", repeats=1)

        self.assertEqual(len(rows), 2)
        self.assertEqual(meta["metrics"]["case_pass_rate"], 0.5)
        self.assertEqual(meta["metrics"]["eval_failure_rate"], 0.5)
        self.assertEqual(meta["failure_breakdown"], {"contract_violation": 1})

    def test_mismatch_row_keeps_actual_memory_and_supersedes_id(self):
        case = next(case for case in CASES if case["id"] == "same_topic_additional_requirement")
        row = score_case(case, outcome({
            "action": "update",
            "memory": "Research summaries should be detailed.",
            "supersedes_id": "m1",
            "reason": "wrong action",
        }), 1)
        self.assertEqual(row["failure_type"], "memory_decision_mismatch")
        self.assertEqual(row["actual_memory"], "Research summaries should be detailed.")
        self.assertEqual(row["actual_supersedes_id"], "m1")

    def test_write_update_and_ignore_decisions(self):
        expected = {
            "new_language_preference": "write",
            "update_concise_to_detailed": "update",
            "duplicate_concise_preference": "ignore",
            "task_only_chinese": "ignore",
            "explicit_no_memory": "ignore",
            "hedged_future_preference": "ignore",
        }
        rows, _ = run_eval(provider="fixture", repeats=1)
        actual = {row["case"]: row["actual_action"] for row in rows}
        for case_id, action in expected.items():
            self.assertEqual(actual[case_id], action)

    def test_concept_alternative_paraphrase_passes(self):
        case = next(case for case in CASES if case["id"] == "new_language_preference")
        parsed = {
            "action": "write",
            "memory": "Project docs should use Chinese.",
            "supersedes_id": None,
            "reason": "paraphrase",
        }
        self.assertTrue(score_case(case, outcome(parsed), 1)["case_pass"])

    def test_required_table_constraint_cannot_be_omitted(self):
        case = next(case for case in CASES if case["id"] == "new_research_report_constraint")
        parsed = {
            "action": "write",
            "memory": "Factor reports should include assumptions and limitations.",
            "supersedes_id": None,
            "reason": "missing table concept",
        }
        row = score_case(case, outcome(parsed), 1)
        self.assertFalse(row["memory_content_correct"])
        self.assertFalse(row["case_pass"])

    def test_invalid_supersedes_id_is_contract_violation(self):
        case = next(case for case in CASES if case["id"] == "update_concise_to_detailed")
        parsed = {
            "action": "update",
            "memory": "Detailed research explanations with assumptions.",
            "supersedes_id": "missing",
            "reason": "bad fixture",
        }
        row = score_case(case, run_case(case, client=StaticClient(parsed), model="fixture"), 1)
        self.assertEqual(row["eval_status"], "human_review")
        self.assertEqual(row["failure_type"], "contract_violation")

    def test_shape_and_action_errors_are_contract_violations(self):
        case = CASES[0]
        for parsed in (
            {"action": "write"},
            {"action": "remember", "memory": "x", "supersedes_id": None, "reason": "bad"},
        ):
            row = score_case(case, run_case(case, client=StaticClient(parsed), model="fixture"), 1)
            self.assertEqual(row["failure_type"], "contract_violation")

    def test_malformed_json_and_provider_error_are_attributed(self):
        case = CASES[0]
        malformed = score_case(case, run_case(case, client=StaticClient("not json"), model="fixture"), 1)
        self.assertEqual(malformed["failure_type"], "malformed_response")
        provider = score_case(case, run_case(case, client=StaticClient(error=RuntimeError("offline")), model="fixture"), 1)
        self.assertEqual(provider["failure_type"], "provider_error")

    def test_fixture_does_not_need_oracle_in_prompt(self):
        case = CASES[0]
        client = FixtureClient(case)
        payload = _prompt(case)
        response = json.loads(client.create(payload)["output_text"])
        self.assertNotIn("expected", payload["input"])
        self.assertEqual(response["action"], "write")


class StaticClient:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error

    def create(self, payload):
        if self.error:
            raise self.error
        text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return {"output_text": text}


if __name__ == "__main__":
    unittest.main()
