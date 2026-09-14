from __future__ import annotations

from unittest.mock import patch
import unittest

from agent.core.contracts import ResearchRun, ToolResult
from agent.tools.executor import execute_steps


def fake_tool(name: str, calls: list, status: str = "success"):
    def execute(*, run_id, **arguments):
        calls.append((name, arguments, run_id))
        return ToolResult(tool_name=name, run_id=run_id, status=status)

    return execute


class MultiStepExecutorTests(unittest.TestCase):
    def test_success_steps_are_ordered_and_share_run_id(self):
        calls = []
        functions = {
            "first": fake_tool("first", calls),
            "second": fake_tool("second", calls),
        }

        with patch("agent.tools.executor.TOOL_FUNCTIONS", functions):
            run = execute_steps([
                {"name": "first", "arguments": {"value": 1}},
                {"name": "second", "arguments": {"value": 2}},
            ], user_request="two steps")

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.final_status, "success")
        self.assertEqual([step["seq"] for step in run.steps], [1, 2])
        self.assertEqual([call[0] for call in calls], ["first", "second"])
        self.assertEqual({call[2] for call in calls}, {run.run_id})

    def test_error_step_is_recorded_and_stops_following_steps(self):
        calls = []
        functions = {
            "first": fake_tool("first", calls),
            "bad": fake_tool("bad", calls, status="error"),
            "never": fake_tool("never", calls),
        }

        with patch("agent.tools.executor.TOOL_FUNCTIONS", functions):
            run = execute_steps([
                {"name": "first", "arguments": {}},
                {"name": "bad", "arguments": {}},
                {"name": "never", "arguments": {}},
            ])

        self.assertEqual(run.status, "failed")
        self.assertEqual(run.final_status, "error")
        self.assertEqual([step["status"] for step in run.steps], ["success", "error"])
        self.assertEqual([call[0] for call in calls], ["first", "bad"])

    def test_partial_then_success_completes_as_partial(self):
        calls = []
        functions = {
            "partial": fake_tool("partial", calls, status="partial"),
            "success": fake_tool("success", calls),
        }

        with patch("agent.tools.executor.TOOL_FUNCTIONS", functions):
            run = execute_steps([
                {"name": "partial", "arguments": {}},
                {"name": "success", "arguments": {}},
            ])

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.final_status, "partial")
        self.assertEqual([call[0] for call in calls], ["partial", "success"])

    def test_existing_running_run_continues_sequence(self):
        calls = []
        run = ResearchRun(run_id="run-1")
        run.add_step(ToolResult(tool_name="old", run_id="run-1"))

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {"next": fake_tool("next", calls)}):
            execute_steps([{"name": "next", "arguments": {}}], run=run)

        self.assertEqual([step["seq"] for step in run.steps], [1, 2])
        self.assertEqual(calls[0][2], "run-1")

    def test_existing_partial_outcome_is_preserved_on_continuation(self):
        calls = []
        run = ResearchRun(run_id="run-1")
        run.add_step(ToolResult(tool_name="partial", run_id="run-1", status="partial"))

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {"next": fake_tool("next", calls)}):
            execute_steps([{"name": "next", "arguments": {}}], run=run)

        self.assertEqual([step["seq"] for step in run.steps], [1, 2])
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.final_status, "partial")

    def test_terminal_run_is_rejected_before_execution(self):
        calls = []
        run = ResearchRun()
        run.complete()

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {"tool": fake_tool("tool", calls)}):
            with self.assertRaises(RuntimeError):
                execute_steps([{"name": "tool", "arguments": {}}], run=run)

        self.assertEqual(calls, [])

    def test_empty_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            execute_steps([])

    def test_tool_argument_error_uses_tool_result_error_contract(self):
        calls = []

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {"tool": fake_tool("tool", calls)}):
            run = execute_steps([{"name": "tool", "arguments": "invalid"}])

        self.assertEqual(run.status, "failed")
        self.assertEqual(run.final_status, "error")
        self.assertEqual(run.steps[0]["status"], "error")
        self.assertEqual(run.steps[0]["errors"][0]["code"], "step_execution_error")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
