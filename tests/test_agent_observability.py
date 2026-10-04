from __future__ import annotations

import json
from datetime import datetime
import unittest
from unittest.mock import Mock, patch

from agent.core.providers import DeepSeekChatClient, OpenAIResponsesClient
from agent.core.safety import SafetyClient
from agent.core.telemetry import (
    BEIJING_TIMEZONE,
    RunTelemetry,
    TelemetryClient,
    _provider_name,
    estimate_cost,
)


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


class ObservabilityTests(unittest.TestCase):
    USAGE = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }

    def test_provider_discovery_does_not_expand_bare_mock(self):
        client = Mock()
        # Fail immediately on dynamic discovery rather than risking another OOM.
        with patch.object(Mock, "__getattr__", side_effect=AssertionError("dynamic attribute lookup")):
            self.assertIsNone(_provider_name(client))
        self.assertEqual(client._mock_children, {})

    def test_provider_discovery_preserves_real_adapters_and_wrappers(self):
        for client, expected in ((DeepSeekChatClient(api_key="fixture"), "deepseek"),
                                 (OpenAIResponsesClient(api_key="fixture"), "openai"),
                                 (Client({}), "fixture"), (Mock(provider="fixture"), "fixture")):
            for wrapped in (client, SafetyClient(client),
                            TelemetryClient(client, RunTelemetry(), stage="planning"),
                            SafetyClient(TelemetryClient(SafetyClient(client), RunTelemetry(), stage="planning"))):
                with self.subTest(provider=expected, wrapper=type(wrapped).__name__):
                    self.assertEqual(_provider_name(wrapped), expected)

    def test_provider_discovery_does_not_evaluate_properties_and_handles_cycles(self):
        class DynamicClient:
            @property
            def provider(self):
                raise AssertionError("provider property evaluated")

            @property
            def client(self):
                raise AssertionError("client property evaluated")

        self.assertIsNone(_provider_name(DynamicClient()))
        cyclic = Mock()
        cyclic.client = cyclic
        self.assertIsNone(_provider_name(cyclic))
        self.assertEqual(cyclic._mock_children, {})

    def test_usage_extraction_and_unknown_cost(self):
        telemetry = RunTelemetry()
        client = Client("{}", self.USAGE)
        payload = {
            "model": "fixture-model",
            "instructions": "return JSON",
            "input": "request",
        }
        TelemetryClient(client, telemetry, stage="planning", model="fixture-model").create(payload)

        self.assertEqual(client.calls, [payload])
        call = telemetry.envelope()["calls"][0]
        self.assertEqual(call["input_tokens"], 10)
        self.assertEqual(call["output_tokens"], 4)
        self.assertEqual(call["cached_tokens"], 2)
        self.assertEqual(call["reasoning_tokens"], 1)
        self.assertIsNone(call["estimated_cost"])
        self.assertIsNone(estimate_cost("unknown", "model", 10, 4))
        self.assertIsNone(estimate_cost("openai", "gpt-5", 10, 4))
        self.assertEqual(estimate_cost("openai", "gpt-5", 10, 4, 2), 0.00005025)
        self.assertIsNone(estimate_cost("deepseek", "unknown", 10, 4, 2))

    def test_deepseek_pricing_uses_beijing_peak_and_cache_tiers(self):
        peak = datetime(2026, 9, 18, 10, tzinfo=BEIJING_TIMEZONE)
        idle = datetime(2026, 9, 18, 13, tzinfo=BEIJING_TIMEZONE)
        self.assertEqual(
            estimate_cost("deepseek", "deepseek-flash", 1_000_000, 1_000_000, 1_000_000, peak),
            8.04,
        )
        self.assertEqual(
            estimate_cost("deepseek", "deepseek-v4-flash", 1_000_000, 1_000_000, 0, idle),
            5.0,
        )
        self.assertEqual(
            estimate_cost("deepseek", "deepseek-v4-flash-vision-exp", 1_000_000, 1_000_000, 1_000_000, idle),
            4.02,
        )
        self.assertEqual(
            estimate_cost("deepseek", "deepseek-v4-pro", 1_000_000, 1_000_000, 0, peak),
            36.0,
        )

    def test_mixed_provider_currencies_never_form_one_cost(self):
        telemetry = RunTelemetry()
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                 "prompt_tokens_details": {"cached_tokens": 0}}
        telemetry.record(
            stage="planning", provider="deepseek", model="deepseek-flash",
            response={"usage": usage}, latency_ms=1, success=True,
            at=datetime(2026, 9, 18, 10, tzinfo=BEIJING_TIMEZONE),
        )
        telemetry.record(
            stage="grounding", provider="openai", model="gpt-5",
            response={"usage": usage}, latency_ms=1, success=True,
        )
        summary = telemetry.envelope()["summary"]
        self.assertIsNone(summary["estimated_cost"])
        self.assertIsNone(summary["estimated_cost_currency"])
        self.assertIsNotNone(telemetry.envelope()["summary"]["per_stage"]["planning"]["estimated_cost"])
        self.assertIsNotNone(telemetry.envelope()["summary"]["per_stage"]["grounding"]["estimated_cost"])


if __name__ == "__main__":
    unittest.main()
