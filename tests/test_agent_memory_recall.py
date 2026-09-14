from __future__ import annotations

import json
import unittest

from agent.memory.recall_eval import (
    CASES,
    FixtureClient,
    _prompt,
    parse_recall_response,
    run_case,
    run_eval,
    score_case,
)
from agent.memory.store import active_memories, apply_memory_decision


class StaticClient:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error

    def create(self, payload):
        if self.error:
            raise self.error
        text = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return {"output_text": text}


def outcome(selected_ids):
    parsed = {"selected_ids": selected_ids, "reason": "test"}
    return {
        "response_text": json.dumps(parsed),
        "parsed": parsed,
        "parse_error": None,
        "contract_error": None,
    }


class MemoryRecallTests(unittest.TestCase):
    def test_dataset_schema_and_expected_ids(self):
        self.assertGreaterEqual(len(CASES), 14)
        self.assertLessEqual(len(CASES), 18)
        for case in CASES:
            self.assertIsInstance(case["id"], str)
            self.assertIsInstance(case["user_request"], str)
            self.assertIsInstance(case["active_memories"], list)
            self.assertIsInstance(case["expected_selected_ids"], list)
            memory_ids = [memory["id"] for memory in case["active_memories"]]
            self.assertEqual(len(memory_ids), len(set(memory_ids)))
            self.assertEqual(
                case["expected_selected_ids"],
                [memory_id for memory_id in memory_ids if memory_id in case["expected_selected_ids"]],
            )

    def test_prompt_contains_only_runtime_input(self):
        case = CASES[0]
        payload = _prompt(case)
        body = json.loads(payload["input"])
        self.assertEqual(set(body), {"user_request", "active_memories"})
        serialized = json.dumps(body, ensure_ascii=False)
        for oracle in ("expected_selected_ids", "slice", "difficulty", "fixture"):
            self.assertNotIn(oracle, serialized)
        self.assertNotIn("expected_selected_ids", payload["instructions"])

    def test_parser_rejects_duplicate_unknown_and_wrong_order_ids(self):
        for selected_ids in (["m1", "m1"], ["missing"], ["m2", "m1"]):
            parsed, error = parse_recall_response(
                json.dumps({"selected_ids": selected_ids, "reason": "test"}),
                ["m1", "m2"],
            )
            self.assertIsNotNone(parsed)
            self.assertIsNotNone(error)

    def test_fixture_passes_all_cases_without_oracle_in_prompt(self):
        rows, meta = run_eval(provider="fixture", repeats=1)
        self.assertEqual(len(rows), len(CASES))
        self.assertTrue(all(row["case_pass"] for row in rows))
        self.assertEqual(meta["metrics"]["exact_match_rate"], 1.0)
        self.assertEqual(meta["metrics"]["case_pass_rate"], 1.0)
        payload = _prompt(CASES[0])
        self.assertNotIn(CASES[0]["expected_selected_ids"].__repr__(), payload["input"])
        response = FixtureClient(CASES[0]).create(payload)
        self.assertEqual(
            json.loads(response["output_text"])["selected_ids"],
            CASES[0]["expected_selected_ids"],
        )

    def test_exact_set_metrics_handle_partial_and_empty_selection(self):
        case = next(case for case in CASES if case["expected_selected_ids"] == ["m1", "m2"])
        row = score_case(case, outcome(["m1"]), 1)
        self.assertEqual(row["precision"], 1.0)
        self.assertEqual(row["recall"], 0.5)
        self.assertEqual(row["f1"], 2 / 3)
        self.assertFalse(row["exact_match"])

        negative_case = next(case for case in CASES if not case["expected_selected_ids"])
        empty = score_case(negative_case, outcome([]), 1)
        self.assertEqual((empty["precision"], empty["recall"], empty["f1"]), (1.0, 1.0, 1.0))
        false_positive = score_case(negative_case, outcome([negative_case["active_memories"][0]["id"]]), 1)
        self.assertEqual((false_positive["precision"], false_positive["recall"], false_positive["f1"]), (0.0, 0.0, 0.0))

    def test_malformed_provider_and_contract_failures_are_attributed(self):
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
                client=StaticClient({"selected_ids": ["missing"], "reason": "bad"}),
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

    def test_store_active_view_excludes_superseded_recall_candidate(self):
        records = apply_memory_decision(
            [],
            {"action": "write", "memory": "Use concise research summaries.", "supersedes_id": None},
            new_id="m1",
        )
        records = apply_memory_decision(
            records,
            {"action": "update", "memory": "Use detailed research summaries.", "supersedes_id": "m1"},
            new_id="m2",
        )
        active = active_memories(records)
        self.assertEqual([memory["id"] for memory in active], ["m2"])
        body = json.loads(_prompt({
            "user_request": "Write a research summary.",
            "active_memories": list(active),
        })["input"])
        self.assertEqual([memory["id"] for memory in body["active_memories"]], ["m2"])


if __name__ == "__main__":
    unittest.main()
