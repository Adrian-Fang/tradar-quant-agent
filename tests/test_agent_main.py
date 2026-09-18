import json
import os
import unittest
from unittest.mock import Mock, patch

from agent.main import (
    create_provider_client,
    format_human,
    format_json,
    run_request,
)


class AgentMainTests(unittest.TestCase):
    def test_provider_factory_selects_existing_adapters(self):
        with patch("agent.main.DeepSeekChatClient") as deepseek:
            client = create_provider_client("deepseek", api_key="key")
            self.assertIs(client, deepseek.return_value)
            deepseek.assert_called_once_with(api_key="key")

        with patch("agent.main.OpenAIResponsesClient") as openai:
            client = create_provider_client("openai", api_key="key")
            self.assertIs(client, openai.return_value)
            openai.assert_called_once_with(api_key="key")

    def test_provider_and_missing_configuration_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            create_provider_client("other")
        with self.assertRaises(ValueError):
            run_request("test", provider="fixture")

        with patch("agent.main.DeepSeekChatClient", side_effect=RuntimeError("missing key")):
            with self.assertRaisesRegex(RuntimeError, "missing key"):
                run_request("test", provider="deepseek")

    def test_no_retrieval_reuses_one_client_without_embedding(self):
        client = Mock()
        expected = {"status": "ok"}
        with patch("agent.main.run_agent", return_value=expected) as run:
            result = run_request("test", provider="fixture", client=client)

        self.assertIs(result, expected)
        kwargs = run.call_args.kwargs
        for name in (
            "planner_client",
            "hitl_client",
            "synthesis_client",
            "grounding_client",
        ):
            self.assertIs(kwargs[name], client)
        self.assertIsNone(kwargs["retrieval_client"])
        self.assertIsNone(kwargs["semantic_embedder"])
        self.assertEqual(kwargs["model"], "")
        self.assertEqual(
            kwargs["product_boundaries"],
            ["Tradar does not execute live trading or place real-money orders."],
        )

    def test_explicit_product_boundaries_are_preserved(self):
        client = Mock()
        boundaries = ["Custom product boundary."]
        with patch("agent.main.run_agent", return_value={"status": "ok"}) as run:
            run_request(
                "test",
                provider="fixture",
                client=client,
                product_boundaries=boundaries,
            )
        self.assertIs(run.call_args.kwargs["product_boundaries"], boundaries)

    def test_default_boundary_blocks_live_trade_before_model_or_tools(self):
        client = Mock()
        result = run_request(
            "Execute live trading for the requested stock.",
            provider="fixture",
            client=client,
        )

        self.assertEqual(result["observed"]["outcome"], {"status": "blocked"})
        self.assertEqual(result["safety"]["rule"], "product_boundary")
        client.create.assert_not_called()

    def test_retrieval_uses_same_chat_client_and_selected_embedder(self):
        client = Mock()
        embedder = Mock()
        with patch("agent.main.run_agent", return_value={"status": "ok"}) as run:
            run_request(
                "test",
                provider="fixture",
                client=client,
                retrieval="ollama",
                embedding_client=embedder,
            )

        kwargs = run.call_args.kwargs
        self.assertIs(kwargs["retrieval_client"], client)
        self.assertIs(kwargs["semantic_embedder"], embedder)

    def test_retrieval_client_is_constructed_only_when_enabled(self):
        with patch("agent.main.run_agent", return_value={"status": "ok"}), patch(
            "agent.main.OllamaEmbeddingClient"
        ) as ollama:
            run_request("test", provider="fixture", client=Mock(), retrieval="ollama")
        ollama.assert_called_once_with()

    def test_cli_formats_compact_human_and_json_outputs(self):
        result = {
            "status": "ok",
            "answer": "A supported answer.",
            "observed": {"outcome": {"status": "success"}},
            "telemetry": {"summary": {
                "calls": 4,
                "total_tokens": 123,
                "estimated_cost": 0.01,
                "estimated_cost_currency": "CNY",
                "wall_clock_ms": 42.5,
                "failure_stage": None,
                "terminal_stage": "grounding",
            }},
        }
        text = format_human(result)
        self.assertIn("status: ok", text)
        self.assertIn("calls=4", text)
        self.assertIn("estimated_cost=0.01 CNY", text)
        self.assertNotIn('"observed"', text)
        self.assertEqual(json.loads(format_json(result))["answer"], "A supported answer.")

    def test_model_defaults_follow_environment_conventions(self):
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "deepseek-flash"}), patch(
            "agent.main.run_agent", return_value={"status": "ok"}
        ) as run:
            run_request("test", provider="deepseek", client=Mock())
        self.assertEqual(run.call_args.kwargs["model"], "deepseek-flash")

        with patch.dict(os.environ, {}, clear=True), patch(
            "agent.main.run_agent", return_value={"status": "ok"}
        ) as run:
            run_request("test", provider="deepseek", client=Mock())
        self.assertEqual(run.call_args.kwargs["model"], "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
