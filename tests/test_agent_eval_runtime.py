from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.agent_eval import RUNTIME_CASES, run_eval, run_runtime_case, score_case
from agent.core.contracts import ResearchRun, ToolResult


class AgentRuntimeEvalTests(unittest.TestCase):
    def test_runtime_dataset_has_runtime_inputs_without_observed_oracle_trace(self):
        self.assertEqual(len(RUNTIME_CASES), 8)
        for case in RUNTIME_CASES:
            self.assertEqual(set(case), {"id", "slice", "request", "runtime", "expected"})
            self.assertNotIn("observed", case)
            self.assertIn("proposed_action", case["runtime"])

    def test_runtime_case_uses_run_agent_and_existing_scorer(self):
        case = next(case for case in RUNTIME_CASES if case["id"] == "runtime_read_only_success")

        def execute(steps, *, run=None, user_request=""):
            run = run or ResearchRun(user_request=user_request)
            for step in steps:
                expected = next(
                    item for item in case["expected"]["execution_steps"]
                    if item["name"] == step["name"]
                )
                run.add_step(ToolResult(
                    tool_name=step["name"],
                    run_id=run.run_id,
                    normalized_args=expected["arguments"],
                ))
            run.complete(final_status="success")
            return run

        with patch("agent.agent.execute_steps", side_effect=execute) as executor:
            result = run_runtime_case(case, provider="fixture")

        row = score_case(case, result["observed"])
        self.assertTrue(row["behavior_pass"])
        self.assertTrue(row["case_pass"])
        executor.assert_called_once()
        self.assertEqual(result["observed"]["outcome"]["status"], "success")

    def test_provider_without_key_is_skipped(self):
        for provider, key in (("deepseek", "DEEPSEEK_API_KEY"), ("openai", "OPENAI_API_KEY")):
            with self.subTest(provider=provider), patch.dict(os.environ, {key: ""}), patch(
                "agent.agent_eval.DeepSeekChatClient" if provider == "deepseek"
                else "agent.agent_eval.OpenAIResponsesClient"
            ) as client:
                rows, meta = run_eval(provider=provider, repeats=1)

            self.assertEqual(rows, [])
            self.assertEqual(meta["status"], "skipped")
            self.assertIn(key, meta["reason"])
            client.assert_not_called()

    def test_real_provider_path_uses_runtime_cases_and_selected_model(self):
        observed = RUNTIME_CASES[0]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "test-model"}), patch(
            "agent.agent_eval.DeepSeekChatClient"
        ) as client, patch(
            "agent.agent_eval.run_runtime_case",
            return_value={"observed": {
                "context": {"selected_ids": ["request_scope"]},
                "planning": {"status": "ready", "steps": observed["expected"]["required_steps"]},
                "steps": [],
                "retrieval": {"status": "not_used", "research_ids": []},
                "research_run": None,
                "grounding": None,
                "hitl": {"decision": "proceed"},
                "orchestration": None,
                "outcome": {"status": "success"},
            }},
        ) as runtime_case:
            rows, meta = run_eval(provider="deepseek", repeats=1)

        client.assert_called_once()
        self.assertEqual(meta["cases"], len(RUNTIME_CASES))
        self.assertEqual(len(rows), len(RUNTIME_CASES))
        self.assertTrue(all(call.kwargs["provider"] == "deepseek" for call in runtime_case.call_args_list))
        self.assertTrue(all(call.kwargs["model"] == "test-model" for call in runtime_case.call_args_list))


if __name__ == "__main__":
    unittest.main()
