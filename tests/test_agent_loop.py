from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.agent import run_agent
from agent.core.contracts import ToolResult
from agent.loop.runner import run_loop


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


class ToolClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        return {"output_text": json.dumps(self.output)}


class SequenceClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        return {"output_text": json.dumps(output)}


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

    def test_agent_delegates_iterative_execution_and_keeps_output_binding(self):
        calls = []
        planner = ToolClient({
            "status": "ready",
            "steps": [step()],
            "reason": "start with universe inspection",
        })
        hitl = ToolClient({
            "decision": "proceed",
            "approval_request": None,
            "reason": "allowed",
        })
        synthesis = ToolClient({
            "status": "success",
            "answer": "Both observations were collected.",
            "evidence_ids": ["step-1-inspect_universe", "step-2-evaluate_factor"],
        })
        grounding = ToolClient({
            "answer": "ignored",
            "claims": [{
                "claim": "Both observations were collected.",
                "evidence_ids": ["step-1-inspect_universe", "step-2-evaluate_factor"],
                "grounding": "supported",
            }],
        })

        def decide(observation):
            if len(observation["observations"]) == 1:
                return execute_decision(step("evaluate_factor"))
            return {"status": "finish", "step": None, "reason": "sufficient"}

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
            "evaluate_factor": fake_tool("evaluate_factor", calls),
        }):
            result = run_agent(
                "Inspect then evaluate.",
                planner_client=planner,
                hitl_client=hitl,
                synthesis_client=synthesis,
                grounding_client=grounding,
                loop_decider=decide,
                run_id="loop-run-1",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["outcome"], {"status": "success"})
        self.assertEqual(result["research_run"].run_id, "loop-run-1")
        self.assertEqual([item["seq"] for item in result["research_run"].steps], [1, 2])
        self.assertEqual(result["observed"]["loop"]["iterations"], 3)
        self.assertEqual(result["answer"], "Both observations were collected.")
        self.assertEqual(len(synthesis.calls), 1)

    def test_default_planner_path_decides_next_step_from_observation(self):
        calls = []
        planner = SequenceClient([
            {
                "status": "ready",
                "steps": [step()],
                "reason": "inspect first",
            },
            {
                "status": "ready",
                "steps": [step("evaluate_factor")],
                "reason": "evaluate the observed universe",
            },
            {"status": "finish", "steps": [], "reason": "enough evidence"},
        ])
        hitl = ToolClient({
            "decision": "proceed",
            "approval_request": None,
            "reason": "allowed",
        })
        synthesis = ToolClient({
            "status": "success",
            "answer": "Both observations were collected.",
            "evidence_ids": ["step-1-inspect_universe", "step-2-evaluate_factor"],
        })
        grounding = ToolClient({
            "answer": "ignored",
            "claims": [{
                "claim": "Both observations were collected.",
                "evidence_ids": ["step-1-inspect_universe", "step-2-evaluate_factor"],
                "grounding": "supported",
            }],
        })

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
            "evaluate_factor": fake_tool("evaluate_factor", calls),
        }):
            result = run_agent(
                "Inspect then evaluate.",
                planner_client=planner,
                hitl_client=hitl,
                synthesis_client=synthesis,
                grounding_client=grounding,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [item["tool_name"] for item in result["research_run"].steps],
            ["inspect_universe", "evaluate_factor"],
        )
        self.assertEqual(len(planner.calls), 3)
        first_input = json.loads(planner.calls[0]["input"])
        self.assertNotIn("observations", first_input)
        second_input = json.loads(planner.calls[1]["input"])
        self.assertEqual(len(second_input["observations"]["observations"]), 1)
        self.assertEqual(
            second_input["observations"]["observations"][0]["tool_name"],
            "inspect_universe",
        )

    def test_observation_can_terminate_provisional_plan(self):
        calls = []
        planner = SequenceClient([
            {
                "status": "ready",
                "steps": [step(), step("evaluate_factor")],
                "reason": "provisional two-step plan",
            },
            {"status": "finish", "steps": [], "reason": "first result is sufficient"},
        ])
        hitl = ToolClient({
            "decision": "proceed",
            "approval_request": None,
            "reason": "allowed",
        })
        synthesis = ToolClient({
            "status": "success",
            "answer": "The first observation is sufficient.",
            "evidence_ids": ["step-1-inspect_universe"],
        })
        grounding = ToolClient({
            "answer": "ignored",
            "claims": [{
                "claim": "The first observation is sufficient.",
                "evidence_ids": ["step-1-inspect_universe"],
                "grounding": "supported",
            }],
        })

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
            "evaluate_factor": fake_tool("evaluate_factor", calls),
        }):
            result = run_agent(
                "Inspect then evaluate if needed.",
                planner_client=planner,
                hitl_client=hitl,
                synthesis_client=synthesis,
                grounding_client=grounding,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual([item["tool_name"] for item in result["research_run"].steps], [
            "inspect_universe",
        ])
        self.assertEqual([call[0] for call in calls], ["inspect_universe"])
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(len(result["observed"]["planning"]["steps"]), 2)

    def test_each_loop_action_is_gated_before_execution(self):
        calls = []
        planner = SequenceClient([
            {
                "status": "ready",
                "steps": [step(), step("evaluate_factor")],
                "reason": "provisional plan",
            },
            {
                "status": "ready",
                "steps": [step("evaluate_factor")],
                "reason": "continue after observation",
            },
        ])
        hitl = SequenceClient([
            {"decision": "proceed", "approval_request": None, "reason": "first"},
            {
                "decision": "needs_approval",
                "approval_request": "Approve the second research action.",
                "reason": "second action needs approval",
            },
        ])

        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": fake_tool("inspect_universe", calls),
            "evaluate_factor": fake_tool("evaluate_factor", calls),
        }):
            result = run_agent(
                "Inspect then evaluate.",
                planner_client=planner,
                hitl_client=hitl,
                synthesis_client=ToolClient({
                    "status": "success",
                    "answer": "first result",
                    "evidence_ids": ["step-1-inspect_universe"],
                }),
                grounding_client=ToolClient({
                    "answer": "ignored",
                    "claims": [{
                        "claim": "first result",
                        "evidence_ids": ["step-1-inspect_universe"],
                        "grounding": "supported",
                    }],
                }),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["outcome"], {"status": "needs_approval"})
        self.assertEqual(result["observed"]["hitl"]["decision"], "needs_approval")
        self.assertEqual([item["tool_name"] for item in result["research_run"].steps], [
            "inspect_universe",
        ])
        self.assertEqual([item[0] for item in calls], ["inspect_universe"])
        self.assertEqual(len(hitl.calls), 2)


if __name__ == "__main__":
    unittest.main()
