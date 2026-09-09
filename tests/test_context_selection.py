from __future__ import annotations

import unittest

from agent.context.selection import select_context
from agent.core.resources import load_json


SELECTION_CASES = load_json("eval/context_selection.json")


class ContextSelectionTests(unittest.TestCase):
    def test_fixture_cases_have_expected_selection(self):
        self.assertEqual(len(SELECTION_CASES), 6)
        for case in SELECTION_CASES:
            result = select_context(case["items"])
            self.assertEqual(
                [item["id"] for item in result["selected"]],
                case["expected_selected"],
                case["id"],
            )
            self.assertEqual(result["dropped"], case["expected_dropped"], case["id"])

    def test_system_rules_are_outside_selectable_context(self):
        with self.assertRaises(ValueError):
            select_context([{"id": "system", "kind": "system", "text": "always present"}])
        with self.assertRaises(ValueError):
            select_context([
                {"id": "current", "kind": "current_instruction", "key": "scope", "text": "current"},
                {"id": "system", "kind": "system", "key": "scope", "text": "always present"},
            ])


if __name__ == "__main__":
    unittest.main()
