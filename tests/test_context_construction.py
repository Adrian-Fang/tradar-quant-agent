from __future__ import annotations

import unittest

from agent.context.construction import construct_context
from agent.core.resources import load_json


CASES = load_json("eval/context_construction.json")


class ContextConstructionTests(unittest.TestCase):
    def test_fixture_cases(self):
        self.assertEqual(len(CASES), 5)
        for case in CASES:
            result = construct_context(
                case["user_request"],
                case["compaction_result"],
                system_rules=case["system_rules"],
                product_rules=case["product_rules"],
            )
            expected = case["expected"]
            self.assertEqual(result["status"], expected["status"], case["id"])
            if expected["status"] != "ok":
                self.assertIsNone(result["instructions"], case["id"])
                self.assertIsNone(result["input"], case["id"])
                continue

            self.assertEqual(
                [item["id"] for item in result["context"]],
                expected["context_order"],
                case["id"],
            )
            self.assertEqual(result["instructions"], expected["instructions"], case["id"])
            for text in expected["input_contains"]:
                self.assertIn(text, result["input"], case["id"])
            for text in expected["input_excludes"]:
                self.assertNotIn(text, result["input"], case["id"])


if __name__ == "__main__":
    unittest.main()
