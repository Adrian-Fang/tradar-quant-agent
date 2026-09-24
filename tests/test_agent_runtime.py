from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import patch

from agent.agent_eval import CASES, score_case
from agent.agent import run_agent
from agent.core.contracts import ToolResult


def response(value):
    return {"output_text": json.dumps(value)}


def plan(status="ready", steps=None):
    return {"status": status, "steps": steps or [], "reason": "test"}


def research_step():
    return {
        "name": "inspect_universe",
        "arguments": {"start_date": "2026-08-31", "end_date": "2026-08-31"},
    }


class Client:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return response(self.output)


def tool(calls, status="success"):
    def execute(*, run_id, **arguments):
        calls.append((arguments, run_id))
        return ToolResult(tool_name="inspect_universe", run_id=run_id, status=status)

    return execute


class AgentRuntimeTests(unittest.TestCase):
    def clients(self, planner_output=None, gate="proceed", grounding=None):
        planner = Client(planner_output or plan(steps=[research_step()]))
        hitl = Client({
            "decision": gate,
            "approval_request": "Approve the action." if gate == "needs_approval" else None,
            "reason": "test",
        })
        ground = Client(grounding) if grounding is not None else None
        return planner, hitl, ground

    def run_with_tool(self, *, planner_output=None, gate="proceed", grounding=None, tool_status="success", answer=None):
        planner, hitl, ground = self.clients(planner_output, gate, grounding)
        calls = []
        with patch("agent.tools.executor.TOOL_FUNCTIONS", {
            "inspect_universe": tool(calls, tool_status),
        }):
            result = run_agent(
                "Inspect the universe.",
                planner_client=planner,
                hitl_client=hitl,
                grounding_client=ground,
                answer=answer,
                evidence=[{"id": "e1", "text": "The universe was inspected."}] if answer is not None else None,
            )
        return result, planner, hitl, ground, calls

    def test_success_has_ae09_trace_and_grounding(self):
        result, planner, hitl, ground, calls = self.run_with_tool(
            answer="The universe was inspected.",
            grounding={
                "answer": "ignored",
                "claims": [{
                    "claim": "The universe was inspected.",
                    "evidence_ids": ["e1"],
                    "grounding": "supported",
                }],
            },
        )
        observed = result["observed"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["research_run"].status, "completed")
        self.assertEqual(observed["outcome"], {"status": "success"})
        self.assertEqual(observed["context"], {"selected_ids": ["request_scope"]})
        self.assertEqual(observed["planning"]["status"], "ready")
        self.assertEqual(observed["steps"][0]["status"], "success")
        self.assertTrue(observed["grounding"]["fully_grounded"])
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(len(hitl.calls), 1)
        self.assertEqual(len(ground.calls), 1)
        self.assertEqual(len(calls), 1)

    def test_needs_input_and_no_action_stop_before_gate(self):
        for status in ("needs_input", "no_action"):
            with self.subTest(status=status):
                result, planner, hitl, _, calls = self.run_with_tool(
                    planner_output=plan(status),
                )
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["plan"]["status"], status)
                self.assertIsNone(result["research_run"])
                self.assertIsNone(result["observed"]["hitl"])
                self.assertEqual(hitl.calls, [])
                self.assertEqual(calls, [])

    def test_meta_no_action_returns_grounded_capability_answer_without_tools(self):
        planner, hitl, _, = self.clients(
            planner_output=plan("no_action"),
        )
        result = run_agent(
            "你有哪些数据，数据质量怎么样？",
            planner_client=planner,
            hitl_client=hitl,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["outcome"], {"status": "success"})
        self.assertTrue(result["answer"])
        self.assertIn("没有统一的 data-quality aggregate score", result["answer"])
        self.assertNotIn("质量评分为", result["answer"])
        self.assertIsNone(result["research_run"])
        self.assertEqual(result["observed"]["steps"], [])
        self.assertTrue(result["grounding"]["fully_grounded"])
        self.assertEqual(hitl.calls, [])

    def test_conversation_history_reaches_planner_as_history_items(self):
        planner, hitl, _, = self.clients(
            planner_output=plan("no_action"),
        )
        result = run_agent(
            "那 2026 年呢？",
            planner_client=planner,
            hitl_client=hitl,
            conversation_history=[
                {"role": "user", "content": "研究 high52 在 2025 年的表现。"},
            ],
        )

        self.assertEqual(result["status"], "ok")
        planning_input = json.loads(planner.calls[0]["input"])
        self.assertIn("conversation-1", planning_input["user_request"])
        self.assertIn("high52", planning_input["user_request"])

    def test_planner_error_is_returned_without_execution(self):
        planner, hitl, _, = self.clients()
        planner.error = RuntimeError("offline")
        result = run_agent("Inspect the universe.", planner_client=planner, hitl_client=hitl)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "provider_error")
        self.assertIsNone(result["research_run"])
        self.assertEqual(result["observed"]["planning"]["status"], "error")
        self.assertEqual(result["observed"]["planning"]["error_type"], "provider_error")
        self.assertEqual(hitl.calls, [])

    def test_hitl_needs_approval_stops_before_execution(self):
        for decision in ("needs_approval", "blocked"):
            with self.subTest(decision=decision):
                result, _, hitl, _, calls = self.run_with_tool(gate=decision)

                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["observed"]["hitl"]["decision"], decision)
                self.assertEqual(result["observed"]["outcome"]["status"], decision)
                self.assertIsNone(result["research_run"])
                self.assertIsNone(result["observed"]["grounding"])
                self.assertEqual(len(hitl.calls), 1)
                self.assertEqual(calls, [])

    def test_runtime_hitl_stop_scores_as_controlled_stop(self):
        case = next(case for case in CASES if case["id"] == "hitl_needs_approval")
        result, _, _, _, calls = self.run_with_tool(
            planner_output={
                "status": "ready",
                "steps": case["expected"]["required_steps"],
                "reason": "test",
            },
            gate="needs_approval",
        )

        row = score_case(case, result["observed"])
        self.assertTrue(row["behavior_pass"])
        self.assertTrue(row["diagnostic_pass"])
        self.assertIsNone(row["failure_stage"])
        self.assertIsNone(row["trajectory"])
        self.assertEqual(calls, [])

    def test_tool_error_is_execution_failure_inside_research_run(self):
        result, _, _, _, calls = self.run_with_tool(tool_status="error")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["research_run"].status, "failed")
        self.assertEqual(result["research_run"].final_status, "error")
        self.assertEqual(result["observed"]["steps"][0]["status"], "error")
        self.assertEqual(result["observed"]["outcome"]["status"], "error")
        self.assertEqual(len(calls), 1)

    def test_grounding_failure_blocks_final_outcome_without_rewriting(self):
        result, _, _, ground, _ = self.run_with_tool(
            answer="The universe is permanently reliable.",
            grounding={
                "answer": "ignored",
                "claims": [{
                    "claim": "The universe is permanently reliable.",
                    "evidence_ids": ["e1"],
                    "grounding": "unsupported",
                }],
            },
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["outcome"]["status"], "blocked")
        self.assertFalse(result["observed"]["grounding"]["fully_grounded"])
        self.assertEqual(result["grounding"]["answer"], "The universe is permanently reliable.")
        self.assertEqual(len(ground.calls), 1)

    def test_retrieval_abstain_stops_before_planning(self):
        planner, hitl, _, = self.clients()
        with patch("agent.agent.retrieve_verified", return_value={
            "status": "abstain",
            "results": [],
            "rejected": [],
            "errors": [],
        }) as retrieve:
            result = run_agent(
                "Use research records to answer this question.",
                planner_client=planner,
                hitl_client=hitl,
                retrieval_client=Client(),
                semantic_embedder=object(),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observed"]["retrieval"]["status"], "abstain")
        self.assertEqual(result["observed"]["outcome"]["status"], "abstain")
        self.assertIsNone(result["research_run"])
        self.assertEqual(planner.calls, [])
        retrieve.assert_called_once()

    def test_context_items_are_selected_before_planning(self):
        result, planner, _, _, _ = self.run_with_tool(
            answer="The universe was inspected.",
            grounding={
                "answer": "ignored",
                "claims": [{
                    "claim": "The universe was inspected.",
                    "evidence_ids": ["e1"],
                    "grounding": "supported",
                }],
            },
        )
        body = json.loads(planner.calls[0]["input"])
        self.assertIn("request_scope", body["user_request"])

        extra = copy.deepcopy(result["observed"])
        self.assertEqual(extra["context"]["selected_ids"], ["request_scope"])


if __name__ == "__main__":
    unittest.main()
