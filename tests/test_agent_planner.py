from __future__ import annotations

import json
import unittest

from agent.planning.planner import plan_request


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.payload = None

    def create(self, payload):
        self.payload = payload
        if self.error:
            raise self.error
        return self.response


def response(plan):
    return {"output_text": json.dumps(plan, ensure_ascii=False)}


class RuntimePlannerTests(unittest.TestCase):
    def test_ready_single_step_returns_plan(self):
        client = FakeClient(response({
            "status": "ready",
            "steps": [{"name": "evaluate_factor", "arguments": {
                "factor": "high52",
                "observe_start": "2025-01-01",
                "observe_end": "2025-03-31",
            }}],
            "reason": "enough inputs",
        }))

        result = plan_request("evaluate high52", client=client)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["plan"]["status"], "ready")
        self.assertEqual(result["plan"]["steps"][0]["name"], "evaluate_factor")

    def test_ready_plan_accepts_valid_optional_argument(self):
        result = plan_request("evaluate high52", client=FakeClient(response({
            "status": "ready",
            "steps": [{"name": "evaluate_factor", "arguments": {
                "factor": "high52",
                "observe_start": "2025-01-01",
                "observe_end": "2025-03-31",
                "ic_method": "spearman",
            }}],
            "reason": "enough inputs",
        })))

        self.assertEqual(result["status"], "ok")

    def test_missing_required_argument_is_malformed_response(self):
        result = plan_request("evaluate high52", client=FakeClient(response({
            "status": "ready",
            "steps": [{"name": "evaluate_factor", "arguments": {
                "factor": "high52",
                "observe_start": "2025-01-01",
            }}],
            "reason": "missing date",
        })))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "malformed_response")
        self.assertIn("observe_end", result["error"])

    def test_unknown_argument_is_malformed_response(self):
        result = plan_request("evaluate high52", client=FakeClient(response({
            "status": "ready",
            "steps": [{"name": "evaluate_factor", "arguments": {
                "factor": "high52",
                "observe_start": "2025-01-01",
                "observe_end": "2025-03-31",
                "invented": True,
            }}],
            "reason": "unknown argument",
        })))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "malformed_response")
        self.assertIn("invented", result["error"])

    def test_ready_multi_step_preserves_order(self):
        plan = {
            "status": "ready",
            "steps": [
                {"name": "inspect_universe", "arguments": {"start_date": "2026-08-31", "end_date": "2026-08-31"}},
                {"name": "evaluate_factor", "arguments": {"factor": "high52", "observe_start": "2025-01-01", "observe_end": "2025-03-31"}},
            ],
            "reason": "ordered plan",
        }

        result = plan_request("inspect then evaluate", client=FakeClient(response(plan)))

        self.assertEqual(
            [step["name"] for step in result["plan"]["steps"]],
            ["inspect_universe", "evaluate_factor"],
        )

    def test_needs_input_and_no_action_return_empty_plans(self):
        for status in ("needs_input", "no_action"):
            result = plan_request(
                "request",
                client=FakeClient(response({"status": status, "steps": [], "reason": "stop"})),
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["plan"]["status"], status)
            self.assertEqual(result["plan"]["steps"], [])

    def test_malformed_response_is_error(self):
        result = plan_request("request", client=FakeClient({"output_text": "not json"}))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "malformed_response")
        self.assertIsNone(result["plan"])

    def test_provider_error_is_error(self):
        result = plan_request("request", client=FakeClient(error=RuntimeError("offline")))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "provider_error")
        self.assertIsNone(result["plan"])

    def test_prompt_has_only_request_and_tool_schemas_and_does_not_execute_tools(self):
        client = FakeClient(response({"status": "no_action", "steps": [], "reason": "not applicable"}))

        result = plan_request("recommend a stock", client=client)
        prompt_input = json.loads(client.payload["input"])

        self.assertEqual(set(prompt_input), {"user_request", "tool_schemas"})
        self.assertNotIn("expected_status", client.payload["input"])
        self.assertNotIn("expected_steps", client.payload["input"])
        self.assertEqual(result["status"], "ok")

    def test_delegated_event_study_fixture_uses_canonical_experiment_not_paths(self):
        request = (
            "帮我测一下这个策略：大盘当天跌1%以上时收盘买入A股，"
            "研究2025-01-01到2026-09-23，其他合理假设你自己决定并说明。"
        )
        experiment = {
            "name": "run_research_experiment",
            "arguments": {
                "spec": {
                    "objective": "Estimate forward returns after a broad-market down day.",
                    "method": "event study",
                    "inputs": {
                        "start_date": "2025-01-01",
                        "end_date": "2026-09-23",
                        "market_proxy": "000300",
                        "horizons": [1, 3, 5, 10, 20],
                    },
                    "assumptions": ["Use CSI 300 / 000300 as the broad-market proxy."],
                    "outputs": ["forward returns", "sample counts"],
                },
            },
        }
        client = FakeClient(response({
            "status": "ready",
            "steps": [experiment],
            "reason": "Use canonical local data and explicit default assumptions.",
        }))

        result = plan_request(request, client=client)
        prompt_input = json.loads(client.payload["input"])

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["plan"]["status"], "ready")
        self.assertEqual(result["plan"]["steps"][0]["name"], "run_research_experiment")
        self.assertNotIn("needs_input", result["plan"]["status"])
        self.assertNotIn("weights.csv", json.dumps(prompt_input, ensure_ascii=False))
        self.assertNotIn("program", json.dumps(result["plan"], ensure_ascii=False))
        self.assertNotIn("load_index", client.payload["instructions"])
        self.assertIn("CSI 300 / `000300`", client.payload["instructions"])


if __name__ == "__main__":
    unittest.main()
