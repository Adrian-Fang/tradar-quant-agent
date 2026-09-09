from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.core.contracts import ToolResult
from agent.core.providers import DeepSeekChatClient, OpenAIResponsesClient
from agent.tools.eval import (
    CASES,
    EVAL_ARTIFACTS,
    _prepare_eval_artifacts,
    score_case,
)
from agent.tools.runner import TOOL_SCHEMAS, _request_payload, run_tool_calling
from agent.tools.tools import run_backtest as agent_run_backtest


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def create(self, payload):
        self.payload = payload
        return self.response


class FakeSDK:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.response = response
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return self.response


class ToolCallingTests(unittest.TestCase):
    def test_schemas_are_stable_and_distinct(self):
        self.assertEqual(
            [schema["name"] for schema in TOOL_SCHEMAS],
            ["inspect_universe", "evaluate_factor", "run_backtest"],
        )
        for schema in TOOL_SCHEMAS:
            self.assertEqual(schema["type"], "function")
            self.assertFalse(schema["parameters"]["additionalProperties"])
            self.assertTrue(schema["description"])
        self.assertEqual(
            sorted(TOOL_SCHEMAS[0]["parameters"]["properties"]),
            sorted({"start_date", "end_date", "exclude_st", "min_turnover_rate", "min_listed_days", "snapshot_dates"}),
        )

    def test_llm_call_executes_one_tool_and_records_trace(self):
        response = {
            "output": [{
                "type": "function_call",
                "name": "inspect_universe",
                "arguments": json.dumps({"start_date": "2026-01-01", "end_date": "2026-01-31"}),
            }],
        }
        fake = FakeClient(response)
        executed = ToolResult(
            tool_name="inspect_universe",
            run_id="run-1",
            normalized_args={"start_date": "2026-01-01", "end_date": "2026-01-31"},
            result={"summary": {"eligible": 1}},
        )
        with patch("agent.tools.runner.TOOL_FUNCTIONS", {"inspect_universe": lambda **kwargs: executed}):
            outcome = run_tool_calling("inspect universe", client=fake, run_id="run-1")

        self.assertEqual(outcome["selected_tool"], "inspect_universe")
        self.assertEqual(outcome["tool_result"].status, "success")
        trace = outcome["research_run"]
        self.assertEqual(trace.user_request, "inspect universe")
        self.assertEqual(trace.final_status, "success")
        self.assertEqual(trace.steps[0]["selected_tool"], "inspect_universe")
        self.assertEqual(trace.steps[0]["model_args"]["start_date"], "2026-01-01")
        self.assertEqual(fake.payload["tool_choice"], "required")
        json.loads(trace.to_json())

    def test_multiple_calls_return_structured_error(self):
        fake = FakeClient({
            "output": [
                {"type": "function_call", "name": "inspect_universe", "arguments": "{}"},
                {"type": "function_call", "name": "evaluate_factor", "arguments": "{}"},
            ]
        })
        outcome = run_tool_calling("do both", client=fake)
        self.assertEqual(outcome["tool_result"].status, "error")
        self.assertEqual(outcome["tool_result"].errors[0]["code"], "invalid_model_tool_call")
        self.assertEqual(outcome["research_run"].final_status, "error")

    def test_deepseek_provider_uses_minimal_non_thinking_chat_payload(self):
        response = {
            "choices": [{"message": {"tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "inspect_universe",
                    "arguments": '{"start_date":"2026-01-01","end_date":"2026-01-31"}',
                },
            }]}}],
        }
        fake_sdk = FakeSDK(SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="inspect_universe",
                        arguments='{"start_date":"2026-01-01","end_date":"2026-01-31"}',
                    ),
                ),
            ]))],
        ))
        with patch("agent.core.providers.OpenAI") as openai_factory:
            client = DeepSeekChatClient(api_key="test-key", sdk_client=fake_sdk)
            normalized = client.create(_request_payload("统计 universe", "deepseek-v4-flash"))
            openai_factory.assert_not_called()

        self.assertEqual(normalized["output"][0]["name"], "inspect_universe")
        request_payload = fake_sdk.payload
        self.assertEqual(request_payload["model"], "deepseek-v4-flash")
        self.assertEqual(request_payload["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(request_payload["temperature"], 0.0)
        self.assertEqual(request_payload["messages"][1], {"role": "user", "content": "统计 universe"})
        self.assertEqual(len(request_payload["tools"]), 3)
        self.assertEqual(request_payload["tools"][0]["type"], "function")
        self.assertNotIn("AGENTS.md", json.dumps(request_payload, ensure_ascii=False, default=str))

    def test_deepseek_provider_supports_text_payload_without_tools(self):
        fake_sdk = FakeSDK(SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[], content='{"decision":"ok"}',
            ))],
        ))
        client = DeepSeekChatClient(api_key="test-key", sdk_client=fake_sdk)
        normalized = client.create({
            "model": "deepseek-v4-flash",
            "instructions": "return JSON",
            "input": "context",
        })

        self.assertEqual(normalized["output_text"], '{"decision":"ok"}')
        self.assertNotIn("tools", fake_sdk.payload)
        self.assertNotIn("tool_choice", fake_sdk.payload)

    def test_openai_provider_uses_installed_sdk_resource(self):
        fake_sdk = SimpleNamespace(responses=SimpleNamespace(create=lambda **_payload: {}))
        with patch("agent.core.providers.OpenAI", return_value=fake_sdk) as openai_factory:
            client = OpenAIResponsesClient(api_key="test-key")
        openai_factory.assert_called_once()
        self.assertIs(client.client, fake_sdk)

    def test_benchmark_covers_three_tool_boundaries(self):
        self.assertEqual(len(CASES), 8)
        self.assertEqual(
            {case["expected_tool"] for case in CASES},
            {"inspect_universe", "evaluate_factor", "run_backtest"},
        )
        case = next(case for case in CASES if case["id"] == "strategy_backtest_cost_override")
        row = score_case(
            case,
            {
                "selected_tool": "run_backtest",
                "model_args": case["expected_args"],
                "tool_result": ToolResult(tool_name="run_backtest"),
            },
            repeat=1,
        )
        self.assertTrue(row["selection_ok"])
        self.assertEqual(row["argument_accuracy"], 1.0)
        self.assertTrue(row["execution_ok"])

    def test_fixture_artifacts_are_deterministic_and_readable(self):
        paths = _prepare_eval_artifacts()
        self.assertEqual(paths, EVAL_ARTIFACTS)
        contents = {path: path.read_bytes() for path in paths.values()}
        _prepare_eval_artifacts()
        for path in paths.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), contents[path])

        result = agent_run_backtest(
            str(paths["weights"]),
            str(paths["close"]),
            open_panel=str(paths["open"]),
        )
        self.assertEqual(result.status, "success")


if __name__ == "__main__":
    unittest.main()
