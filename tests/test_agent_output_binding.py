from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.agent import run_agent, tool_results_to_evidence
from agent.core.contracts import ToolResult


def output(value):
    return {"output_text": json.dumps(value)}


def plan(steps):
    return {"status": "ready", "steps": steps, "reason": "fixture"}


def step(number=1):
    dates = {1: "2026-08-31", 2: "2026-09-01"}
    return {
        "name": "inspect_universe",
        "arguments": {
            "start_date": dates[number],
            "end_date": dates[number],
        },
    }


class Client:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return output(self.value)


def tool(status="success"):
    def inspect(*, run_id, **arguments):
        return ToolResult(
            tool_name="inspect_universe",
            run_id=run_id,
            status=status,
            normalized_args=arguments,
            result={
                "as_of": arguments["start_date"],
                "count": 12,
                "direction": "up",
            },
        )

    return inspect


class AgentOutputBindingTests(unittest.TestCase):
    def clients(self, *, steps=None, synthesis=None, grounding=None, tool_status="success"):
        planner = Client(plan(steps or [step()]))
        hitl = Client({"decision": "proceed", "approval_request": None, "reason": "fixture"})
        synthesis_client = Client(synthesis) if synthesis is not None else None
        grounding_client = Client(grounding) if grounding is not None else None
        return planner, hitl, synthesis_client, grounding_client, tool_status

    def run_request(self, *, steps=None, synthesis=None, grounding=None, tool_status="success"):
        planner, hitl, synthesis_client, grounding_client, tool_status = self.clients(
            steps=steps,
            synthesis=synthesis,
            grounding=grounding,
            tool_status=tool_status,
        )
        calls = []
        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": tool(tool_status),
        }):
            result = run_agent(
                "Inspect the universe.",
                planner_client=planner,
                hitl_client=hitl,
                synthesis_client=synthesis_client,
                grounding_client=grounding_client,
                run_id="run-output-binding",
            )
        return result, planner, hitl, synthesis_client, grounding_client, calls

    def supported_grounding(self, evidence_id="step-1-inspect_universe"):
        return {
            "answer": "ignored",
            "claims": [{
                "claim": "The inspected count was 12.",
                "evidence_ids": [evidence_id],
                "grounding": "supported",
            }],
        }

    def test_success_binds_tool_output_then_synthesis_then_grounding(self):
        synthesis = {
            "status": "success",
            "answer": "The inspected count was 12.",
            "evidence_ids": ["step-1-inspect_universe"],
        }
        result, _, _, synth, ground, _ = self.run_request(
            synthesis=synthesis,
            grounding=self.supported_grounding(),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["outcome"], {"status": "success"})
        self.assertEqual(result["answer"], synthesis["answer"])
        self.assertEqual(result["evidence"][0]["id"], "step-1-inspect_universe")
        self.assertEqual(len(synth.calls), 1)
        self.assertEqual(json.loads(synth.calls[0]["input"])["evidence"][0]["text"],
                         '{"as_of":"2026-08-31","count":12,"direction":"up"}')
        self.assertEqual(
            [item["id"] for item in json.loads(ground.calls[0]["input"])["evidence"]],
            ["step-1-inspect_universe"],
        )

    def test_multi_step_evidence_ids_are_stable_and_ordered(self):
        steps = [step(1), step(2)]
        synthesis = {
            "status": "success",
            "answer": "Both inspections returned 12.",
            "evidence_ids": ["step-2-inspect_universe", "step-1-inspect_universe"],
        }
        result, _, _, synth, _, _ = self.run_request(
            steps=steps,
            synthesis=synthesis,
            grounding=self.supported_grounding("step-1-inspect_universe"),
        )
        self.assertEqual(result["research_run"].run_id, "run-output-binding")
        self.assertEqual([step["seq"] for step in result["research_run"].steps], [1, 2])
        evidence = json.loads(synth.calls[0]["input"])["evidence"]
        self.assertEqual([item["id"] for item in evidence], [
            "step-1-inspect_universe", "step-2-inspect_universe",
        ])

    def test_insufficient_synthesis_abstains_without_grounding(self):
        result, _, _, synth, ground, _ = self.run_request(synthesis={
            "status": "insufficient_evidence",
            "answer": "The output is insufficient to answer the request.",
            "evidence_ids": ["step-1-inspect_universe"],
        }, grounding=self.supported_grounding())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["outcome"], {"status": "abstain"})
        self.assertIsNone(result["grounding"])
        self.assertIsNotNone(synth)
        self.assertEqual(ground.calls, [])

    def test_synthesis_provider_and_malformed_errors_fail_closed(self):
        for client in (Client(error=RuntimeError("offline")), Client(value="not json")):
            planner, hitl, _, grounding, _ = self.clients(
                grounding=self.supported_grounding(),
            )
            with patch("agent.tools.executor.TOOL_FUNCTIONS", {
                "inspect_universe": tool(),
            }):
                result = run_agent(
                    "Inspect the universe.",
                    planner_client=planner,
                    hitl_client=hitl,
                    synthesis_client=client,
                    grounding_client=grounding,
                )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_stage"], "synthesis")
            self.assertEqual(
                result["error_type"],
                "provider_error" if client.error else "malformed_response",
            )
            self.assertIsNone(result["grounding"])

    def test_grounding_blocks_after_synthesis(self):
        result, _, _, _, ground, _ = self.run_request(
            synthesis={
                "status": "success",
                "answer": "The universe is permanently reliable.",
                "evidence_ids": ["step-1-inspect_universe"],
            },
            grounding={
                "answer": "ignored",
                "claims": [{
                    "claim": "The universe is permanently reliable.",
                    "evidence_ids": ["step-1-inspect_universe"],
                    "grounding": "unsupported",
                }],
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["outcome"], {"status": "blocked"})
        self.assertFalse(result["grounding"]["fully_grounded"])
        self.assertEqual(len(ground.calls), 1)

    def test_execution_error_skips_synthesis(self):
        planner, hitl, synthesis, grounding, _ = self.clients(
            synthesis={
                "status": "success",
                "answer": "unused",
                "evidence_ids": ["step-1-inspect_universe"],
            },
            grounding=self.supported_grounding(),
            tool_status="error",
        )
        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": tool("error"),
        }):
            result = run_agent(
                "Inspect the universe.",
                planner_client=planner,
                hitl_client=hitl,
                synthesis_client=synthesis,
                grounding_client=grounding,
            )
        self.assertEqual(result["research_run"].status, "failed")
        self.assertEqual(synthesis.calls, [])

    def test_evidence_adapter_uses_only_successful_actual_outputs(self):
        results = [
            ToolResult(
                tool_name="inspect_universe",
                run_id="r",
                status="success",
                result={"value": 10, "date": "2026-08-31"},
            ),
            ToolResult.error("inspect_universe", {}, "failed", "no data", run_id="r"),
        ]
        self.assertEqual(tool_results_to_evidence(results), [{
            "id": "step-1-inspect_universe",
            "text": '{"date":"2026-08-31","value":10}',
        }])


if __name__ == "__main__":
    unittest.main()
