from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.agent import run_agent
from agent.core.contracts import ToolResult
from agent.core.telemetry import RunTelemetry, TelemetryClient, estimate_cost


def response(value, usage=None):
    output = {"output_text": json.dumps(value)}
    if usage is not None:
        output["usage"] = usage
    return output


class Client:
    provider = "fixture"

    def __init__(self, value, usage=None, error=None):
        self.value = value
        self.usage = usage
        self.error = error
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return response(self.value, self.usage)


def research_step():
    return {
        "name": "inspect_universe",
        "arguments": {"start_date": "2026-08-31", "end_date": "2026-08-31"},
    }


class ObservabilityTests(unittest.TestCase):
    USAGE = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }

    def test_usage_extraction_and_unknown_cost(self):
        telemetry = RunTelemetry()
        client = Client("{}", self.USAGE)
        TelemetryClient(client, telemetry, stage="planning", model="fixture-model").create({
            "model": "fixture-model",
            "instructions": "return JSON",
            "input": "request",
        })

        call = telemetry.envelope()["calls"][0]
        self.assertEqual(call["input_tokens"], 10)
        self.assertEqual(call["output_tokens"], 4)
        self.assertEqual(call["cached_tokens"], 2)
        self.assertEqual(call["reasoning_tokens"], 1)
        self.assertIsNone(call["estimated_cost"])
        self.assertIsNone(estimate_cost("unknown", "model", 10, 4))

    def test_run_agent_aggregates_planner_retrieval_hitl_synthesis_grounding(self):
        usage = self.USAGE
        planner = Client({
            "status": "ready",
            "steps": [research_step()],
            "reason": "fixture",
        }, usage)
        retrieval = Client("{}", usage)
        hitl = Client({"decision": "proceed", "approval_request": None, "reason": "fixture"}, usage)
        synthesis = Client({
            "status": "success",
            "answer": "The inspected count was 12.",
            "evidence_ids": ["step-1-inspect_universe"],
        }, usage)
        grounding = Client({
            "answer": "ignored",
            "claims": [{
                "claim": "The inspected count was 12.",
                "evidence_ids": ["step-1-inspect_universe"],
                "grounding": "supported",
            }],
        }, usage)

        def fake_retrieve(query, *, client, model, **kwargs):
            client.create({"model": model, "instructions": "verify", "input": query})
            return {"status": "ok", "results": [], "rejected": [], "errors": []}

        def inspect(*, run_id, **arguments):
            return ToolResult(
                tool_name="inspect_universe",
                run_id=run_id,
                normalized_args=arguments,
                result={"count": 12},
            )

        with patch("agent.agent.retrieve_verified", side_effect=fake_retrieve), patch(
            "agent.tools.executor.TOOL_FUNCTIONS", {"inspect_universe": inspect}
        ):
            result = run_agent(
                "Inspect the universe.",
                planner_client=planner,
                retrieval_client=retrieval,
                semantic_embedder=object(),
                hitl_client=hitl,
                synthesis_client=synthesis,
                grounding_client=grounding,
            )

        telemetry = result["telemetry"]
        self.assertEqual(telemetry["summary"]["calls"], 5)
        self.assertEqual(
            {call["stage"] for call in telemetry["calls"]},
            {"planning", "retrieval_verifier", "hitl", "synthesis", "grounding"},
        )
        self.assertEqual(telemetry["summary"]["input_tokens"], 50)
        self.assertEqual(telemetry["summary"]["output_tokens"], 20)
        self.assertEqual(telemetry["summary"]["cached_tokens"], 10)
        self.assertEqual(telemetry["summary"]["reasoning_tokens"], 5)
        self.assertIsNone(telemetry["summary"]["estimated_cost"])
        self.assertIsNone(telemetry["summary"]["failure_stage"])
        self.assertEqual(set(telemetry["summary"]["per_stage"]), {
            "planning", "retrieval_verifier", "hitl", "synthesis", "grounding",
        })

    def test_provider_failure_is_recorded_with_failure_stage(self):
        planner = Client({}, error=RuntimeError("offline"))
        result = run_agent("Inspect the universe.", planner_client=planner)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["telemetry"]["summary"]["failure_stage"], "planning")
        self.assertFalse(result["telemetry"]["calls"][0]["success"])
        self.assertIsNone(result["telemetry"]["calls"][0]["input_tokens"])


if __name__ == "__main__":
    unittest.main()
