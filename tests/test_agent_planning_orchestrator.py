from __future__ import annotations

import json
from unittest.mock import patch
import unittest

from agent.core.contracts import ToolResult
from agent.planning.orchestrator import run_planned_request


class PlannerClient:
    def __init__(self, plan=None, error=None, output_text=None):
        self.plan = plan
        self.error = error
        self.output_text = output_text
        self.calls = 0

    def create(self, payload):
        self.calls += 1
        if self.error:
            raise self.error
        return {"output_text": self.output_text or json.dumps(self.plan)}


def plan(status="ready", steps=None):
    return {
        "status": status,
        "steps": steps or [],
        "reason": "test plan",
    }


def fake_tool(name, calls, status="success"):
    def execute(*, run_id, **arguments):
        calls.append((name, arguments, run_id))
        return ToolResult(tool_name=name, run_id=run_id, status=status)

    return execute


class PlannedRuntimeTests(unittest.TestCase):
    def test_ready_single_step_executes_and_completes_run(self):
        calls = []
        client = PlannerClient(plan=plan(steps=[{
            "name": "inspect_universe",
            "arguments": {"start_date": "2026-08-31", "end_date": "2026-08-31"},
        }]))

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
        }):
            result = run_planned_request("inspect universe", client=client)

        run = result["research_run"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.final_status, "success")
        self.assertEqual([step["seq"] for step in run.steps], [1])
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.calls, 1)

    def test_ready_multi_step_preserves_order_and_run_id(self):
        calls = []
        client = PlannerClient(plan=plan(steps=[
            {
                "name": "inspect_universe",
                "arguments": {"start_date": "2026-08-31", "end_date": "2026-08-31"},
            },
            {
                "name": "evaluate_factor",
                "arguments": {
                    "factor": "high52",
                    "observe_start": "2025-01-01",
                    "observe_end": "2025-03-31",
                },
            },
        ]))
        functions = {
            "inspect_universe": fake_tool("inspect_universe", calls),
            "evaluate_factor": fake_tool("evaluate_factor", calls),
        }

        with patch("agent.tools.executor.TOOL_FUNCTIONS", functions):
            result = run_planned_request(
                "inspect then evaluate",
                client=client,
                run_id="run-planned-1",
            )

        run = result["research_run"]
        self.assertEqual(run.run_id, "run-planned-1")
        self.assertEqual([step["seq"] for step in run.steps], [1, 2])
        self.assertEqual([call[0] for call in calls], ["inspect_universe", "evaluate_factor"])
        self.assertEqual({call[2] for call in calls}, {"run-planned-1"})
        self.assertEqual(client.calls, 1)

    def test_needs_input_does_not_execute(self):
        calls = []
        client = PlannerClient(plan=plan("needs_input"))

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
        }):
            result = run_planned_request("inspect universe", client=client)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["plan"]["status"], "needs_input")
        self.assertIsNone(result["research_run"])
        self.assertEqual(calls, [])

    def test_no_action_does_not_execute(self):
        calls = []
        client = PlannerClient(plan=plan("no_action"))

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
        }):
            result = run_planned_request("recommend a stock", client=client)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["plan"]["status"], "no_action")
        self.assertIsNone(result["research_run"])
        self.assertEqual(calls, [])

    def test_planner_provider_error_does_not_create_run(self):
        calls = []
        result = run_planned_request(
            "inspect universe",
            client=PlannerClient(error=RuntimeError("offline")),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "provider_error")
        self.assertIsNone(result["plan"])
        self.assertIsNone(result["research_run"])
        self.assertEqual(calls, [])

    def test_planner_malformed_response_does_not_create_run(self):
        result = run_planned_request(
            "inspect universe",
            client=PlannerClient(output_text="not json"),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "malformed_response")
        self.assertIsNone(result["plan"])
        self.assertIsNone(result["research_run"])

    def test_tool_error_stays_in_research_run_and_fails_fast(self):
        calls = []
        client = PlannerClient(plan=plan(steps=[
            {"name": "inspect_universe", "arguments": {
                "start_date": "2026-08-31", "end_date": "2026-08-31",
            }},
            {"name": "evaluate_factor", "arguments": {
                "factor": "high52",
                "observe_start": "2025-01-01",
                "observe_end": "2025-03-31",
            }},
        ]))
        functions = {
            "inspect_universe": fake_tool("inspect_universe", calls, status="error"),
            "evaluate_factor": fake_tool("evaluate_factor", calls),
        }

        with patch("agent.tools.executor.TOOL_FUNCTIONS", functions):
            result = run_planned_request("inspect then evaluate", client=client)

        run = result["research_run"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.final_status, "error")
        self.assertEqual(len(run.steps), 1)
        self.assertEqual([call[0] for call in calls], ["inspect_universe"])
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
