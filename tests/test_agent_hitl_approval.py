from __future__ import annotations

import json
import unittest

from agent.hitl.approval import (
    authorization_from_approval,
    create_approval_request,
    resolve_approval,
)
from agent.hitl.gate import gate_action


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def create(self, payload):
        self.calls += 1
        return {"output_text": self.output}


def action(environment="production"):
    return {
        "description": "Restart the production signal service",
        "environment": environment,
        "reversible": False,
    }


def needs_approval_gate_result():
    return {
        "status": "ok",
        "decision": "needs_approval",
        "approval_request": "Approve restarting the production signal service.",
        "reason": "production change",
        "error_type": None,
        "error": "",
    }


class HitlApprovalTests(unittest.TestCase):
    def test_pending_record_does_not_authorize(self):
        record = create_approval_request(needs_approval_gate_result(), action(), approval_id="a1")
        self.assertEqual(record["status"], "pending")
        self.assertEqual(authorization_from_approval(record, action()), {"approved": False, "scope": None})

    def test_approved_exact_action_authorizes(self):
        record = resolve_approval(
            create_approval_request(needs_approval_gate_result(), action(), approval_id="a1"),
            "approved",
        )
        authorization = authorization_from_approval(record, action())
        self.assertEqual(authorization["approved"], True)
        self.assertEqual(authorization["scope"], record["approval_request"])

    def test_rejected_record_does_not_authorize(self):
        record = resolve_approval(
            create_approval_request(needs_approval_gate_result(), action(), approval_id="a1"),
            "rejected",
        )
        self.assertEqual(authorization_from_approval(record, action()), {"approved": False, "scope": None})

    def test_changed_environment_or_action_scope_does_not_authorize(self):
        record = resolve_approval(
            create_approval_request(needs_approval_gate_result(), action(), approval_id="a1"),
            "approved",
        )
        changed_environment = action("staging")
        changed_scope = {**action(), "description": "Restart every production service"}
        for changed in (changed_environment, changed_scope):
            self.assertEqual(authorization_from_approval(record, changed), {"approved": False, "scope": None})

    def test_terminal_record_cannot_be_resolved_again(self):
        record = resolve_approval(
            create_approval_request(needs_approval_gate_result(), action(), approval_id="a1"),
            "approved",
        )
        with self.assertRaises(ValueError):
            resolve_approval(record, "rejected")

    def test_create_and_resolve_do_not_mutate_input_record(self):
        proposed = action()
        pending = create_approval_request(needs_approval_gate_result(), proposed, approval_id="a1")
        original = dict(pending)
        resolved = resolve_approval(pending, "approved")
        self.assertEqual(pending, original)
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(resolved["status"], "approved")

    def test_gate_to_approval_to_existing_approval_handoff(self):
        client = FakeClient(json.dumps({
            "decision": "needs_approval",
            "approval_request": "Approve restarting the production signal service.",
            "reason": "production change",
        }))
        proposed = action()
        gate_result = gate_action(
            "Restart the production signal service.",
            proposed,
            {"approved": False, "scope": None},
            [],
            client=client,
        )
        record = create_approval_request(gate_result, proposed, approval_id="a1")
        resolved = resolve_approval(record, "approved")
        self.assertEqual(
            authorization_from_approval(resolved, proposed),
            {"approved": True, "scope": gate_result["approval_request"]},
        )
        self.assertEqual(client.calls, 1)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            create_approval_request({"status": "ok", "decision": "proceed"}, action(), approval_id="a1")
        with self.assertRaises(ValueError):
            resolve_approval({"status": "pending"}, "approved")


if __name__ == "__main__":
    unittest.main()
