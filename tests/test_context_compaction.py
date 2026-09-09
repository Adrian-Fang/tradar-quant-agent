from __future__ import annotations

import unittest

from agent.context.compaction import compact_context
from agent.core.resources import load_json


CASES = load_json("eval/context_compaction.json")


class ContextCompactionTests(unittest.TestCase):
    def test_fixture_cases(self):
        self.assertEqual(len(CASES), 7)
        for case in CASES:
            result = compact_context(case["items"], case["budget"])
            expected = case["expected"]
            self.assertEqual(result["status"], expected["status"], case["id"])
            self.assertEqual(
                [item["id"] for item in result["kept"]],
                expected["kept"],
                case["id"],
            )
            self.assertEqual(
                {item["id"]: item["text"] for item in result["compacted"]},
                expected["compacted"],
                case["id"],
            )
            self.assertEqual(result["dropped"], expected["dropped"], case["id"])
            dropped_ids = {item["id"] for item in result["dropped"]}
            self.assertEqual(
                [item["id"] for item in result["context"]],
                [
                    item["id"]
                    for item in case["items"]
                    if item["id"] not in dropped_ids
                ],
                case["id"],
            )

    def test_selected_item_metadata_does_not_trigger_reselection(self):
        item = {
            "id": "history",
            "kind": "history",
            "key": "scope",
            "stale": True,
            "relevant": False,
            "text": "仍然保留这条已选择的上下文。",
        }
        result = compact_context([item], budget=100)
        self.assertEqual(result["kept"], [item])
        self.assertEqual(result["compacted"], [])
        self.assertEqual(result["dropped"], [])

    def test_essential_text_is_not_compacted(self):
        case = next(case for case in CASES if case["id"] == "essential_text_stays_verbatim")
        result = compact_context(case["items"], case["budget"])
        self.assertEqual(
            [item["text"] for item in result["kept"][:2]],
            [case["items"][0]["text"], case["items"][1]["text"]],
        )
        self.assertEqual(result["compacted"], [])


if __name__ == "__main__":
    unittest.main()
