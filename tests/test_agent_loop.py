from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.core.contracts import ToolResult
from agent.loop.runner import compact_tool_observation, run_loop


def step(name="inspect_universe", date="2026-08-31"):
    if name == "evaluate_factor":
        return {
            "name": name,
            "arguments": {
                "factor": "high52",
                "observe_start": "2025-01-01",
                "observe_end": "2025-03-31",
            },
        }
    return {
        "name": name,
        "arguments": {"start_date": date, "end_date": date},
    }


def execute_decision(value):
    return {"status": "execute", "step": value, "reason": "more evidence"}


def fake_tool(name, calls):
    def execute(*, run_id, **arguments):
        calls.append((name, arguments, run_id))
        return ToolResult(
            tool_name=name,
            run_id=run_id,
            normalized_args=arguments,
            result={"tool": name, "sequence": len(calls)},
        )

    return execute


class AgentLoopTests(unittest.TestCase):
    def test_planner_observation_compacts_inspect_membership(self):
        daily_counts = [
            {"date": f"2026-01-{index:02d}", "eligible": index}
            for index in range(1, 31)
        ]
        membership = {
            name: [f"S{index:05d}" for index in range(5000)]
            for name in ("eligible", "trading", "buyable", "sellable")
        }
        full_result = ToolResult(
            tool_name="inspect_universe",
            run_id="large-run",
            normalized_args={"start_date": "2026-01-01", "end_date": "2026-01-30"},
            result={
                "summary": {"symbols": 5000, "trading_days": 30},
                "daily_counts": daily_counts,
                "snapshots": [{
                    "date": "2026-01-15",
                    "is_trading_day": True,
                    "counts": daily_counts[14],
                    "membership": membership,
                }],
            },
        )
        compact = compact_tool_observation(full_result)

        self.assertEqual(compact["tool_name"], "inspect_universe")
        self.assertEqual(compact["status"], "success")
        self.assertEqual(compact["normalized_args"]["start_date"], "2026-01-01")
        self.assertEqual(compact["result"]["summary"]["symbols"], 5000)
        self.assertEqual(compact["result"]["daily_counts"]["omitted_count"], 6)
        self.assertEqual(compact["result"]["snapshots"][0]["date"], "2026-01-15")
        self.assertTrue(compact["result"]["snapshots"][0]["is_trading_day"])
        self.assertEqual(compact["result"]["snapshots"][0]["counts"]["eligible"], 15)
        self.assertEqual(compact["result"]["snapshots"][0]["membership"], {"omitted": True})
        self.assertLess(
            len(json.dumps(compact, ensure_ascii=False)),
            len(json.dumps(full_result.to_dict(), ensure_ascii=False)) / 100,
        )

    def test_loop_passes_compact_observation_after_large_tool_result(self):
        membership = {"eligible": [f"S{index:05d}" for index in range(5000)]}
        seen = []

        def inspect(*, run_id, **arguments):
            return ToolResult(
                tool_name="inspect_universe",
                run_id=run_id,
                normalized_args=arguments,
                result={
                    "summary": {"symbols": 5000},
                    "daily_counts": [{"date": "2026-01-01", "eligible": 5000}],
                    "snapshots": [{"date": "2026-01-01", "membership": membership}],
                },
            )

        def decide(observation):
            if observation["observations"]:
                seen.append(observation)
                return {"status": "finish", "step": None, "reason": "enough"}
            return execute_decision(step())

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {"inspect_universe": inspect}):
            result = run_loop(
                "inspect universe",
                decide_next=decide,
                max_iterations=3,
            )

        self.assertEqual(result["status"], "ok")
        observation = seen[0]["observations"][0]
        self.assertEqual(observation["result"]["summary"]["symbols"], 5000)
        self.assertEqual(observation["result"]["snapshots"][0]["membership"], {"omitted": True})
        self.assertNotIn("S00000", json.dumps(observation, ensure_ascii=False))
        self.assertEqual(result["tool_results"][0].result["snapshots"][0]["membership"]["eligible"][0], "S00000")

    def test_single_tool_then_finish(self):
        calls = []
        decisions = [execute_decision(step()), {
            "status": "finish", "step": None, "reason": "evidence is sufficient",
        }]

        def decide(observation):
            return decisions.pop(0)

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
        }):
            result = run_loop("inspect universe", decide_next=decide, max_iterations=3)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["research_run"].status, "completed")
        self.assertEqual([item["seq"] for item in result["research_run"].steps], [1])
        self.assertEqual(len(calls), 1)

    def test_second_action_is_decided_from_first_observation(self):
        calls = []
        seen = []

        def decide(observation):
            seen.append(observation)
            if len(observation["observations"]) == 0:
                return execute_decision(step())
            if len(observation["observations"]) == 1:
                return execute_decision(step("evaluate_factor"))
            return {"status": "finish", "step": None, "reason": "done"}

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
            "evaluate_factor": fake_tool("evaluate_factor", calls),
        }):
            result = run_loop("inspect then evaluate", decide_next=decide, max_iterations=3)

        self.assertEqual([item["tool_name"] for item in result["research_run"].steps], [
            "inspect_universe", "evaluate_factor",
        ])
        self.assertEqual([item["seq"] for item in result["research_run"].steps], [1, 2])
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(seen[0]["observations"]), 0)
        self.assertEqual(len(seen[1]["observations"]), 1)
        self.assertEqual(len(seen[2]["observations"]), 2)

    def test_controlled_stop_does_not_execute(self):
        calls = []

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
        }):
            result = run_loop(
                "need clarification",
                decide_next=lambda observation: {
                    "status": "needs_input", "step": None, "reason": "missing date",
                },
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["outcome"], "needs_input")
        self.assertIsNone(result["research_run"])
        self.assertEqual(calls, [])

    def test_max_iteration_guard_fails_closed(self):
        calls = []

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
        }):
            result = run_loop(
                "keep working",
                decide_next=lambda observation: execute_decision(step()),
                max_iterations=2,
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "iteration_limit")
        self.assertEqual(result["research_run"].status, "failed")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
