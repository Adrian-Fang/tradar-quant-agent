from __future__ import annotations

import json
import unittest

from agent.hitl.eval import parse_hitl_response as eval_parse_hitl_response
from agent.hitl.gate import gate_action, parse_hitl_response


class FakeClient:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def create(self, payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return {"output_text": self.output}


def response(decision, approval_request=None):
    return json.dumps({
        "decision": decision,
        "approval_request": approval_request,
        "reason": "test",
    })


def action(environment="local", reversible=True):
    return {
        "description": "Edit the application configuration",
        "environment": environment,
        "reversible": reversible,
    }


class HitlGateRuntimeTests(unittest.TestCase):
    def test_proceed_needs_approval_and_blocked_envelopes(self):
        for decision, approval_request in (
            ("proceed", None),
            ("needs_approval", "Approve the production change."),
            ("blocked", None),
        ):
            client = FakeClient(response(decision, approval_request))
            result = gate_action(
                "Perform the proposed action.",
                action("production", False),
                {"approved": False, "scope": None},
                ["Tradar does not execute live trades."] if decision == "blocked" else [],
                client=client,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["decision"], decision)
            self.assertEqual(result["approval_request"], approval_request)
            self.assertIsNone(result["error_type"])
            self.assertEqual(len(client.calls), 1)

    def test_provider_error_fails_closed(self):
        client = FakeClient(error=RuntimeError("offline"))
        result = gate_action(
            "Run the action.", action(), {"approved": False, "scope": None}, [], client=client,
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "provider_error")
        self.assertIsNone(result["decision"])
        self.assertIsNone(result["approval_request"])
        self.assertEqual(len(client.calls), 1)

    def test_malformed_json_and_contract_violation_fail_closed(self):
        for output in (
            "not json",
            json.dumps({"decision": "proceed", "approval_request": "not null", "reason": "bad"}),
        ):
            client = FakeClient(output)
            result = gate_action(
                "Run the action.", action(), {"approved": False, "scope": None}, [], client=client,
            )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_type"], "malformed_response")
            self.assertIsNone(result["decision"])
            self.assertIsNone(result["approval_request"])

    def test_eval_uses_the_runtime_parser(self):
        self.assertIs(eval_parse_hitl_response, parse_hitl_response)

    def test_gate_calls_provider_once_without_retry(self):
        client = FakeClient(response("proceed"))
        result = gate_action(
            "Inspect the repository.", action(), {"approved": False, "scope": None}, [], client=client,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
