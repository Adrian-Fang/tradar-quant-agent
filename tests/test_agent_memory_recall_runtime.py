from __future__ import annotations

import json
import unittest

from agent.memory.recall import recall_memories
from agent.memory.store import active_memories, apply_memory_decision


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


def response(selected_ids):
    return json.dumps({"selected_ids": selected_ids, "reason": "test"})


class MemoryRecallRuntimeTests(unittest.TestCase):
    def test_relevant_selection_returns_active_records_and_calls_once(self):
        records = (
            {"id": "m1", "text": "Use concise reports.", "supersedes_id": None},
            {"id": "m2", "text": "Include source links.", "supersedes_id": None},
        )
        client = FakeClient(response(["m2"]))
        result = recall_memories("Write a report with sources.", records, client=client)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selected_ids"], ["m2"])
        self.assertEqual(result["selected_memories"], [records[1]])
        self.assertEqual(len(client.calls), 1)

    def test_multi_selection_preserves_active_memory_order(self):
        records = (
            {"id": "m1", "text": "First.", "supersedes_id": None},
            {"id": "m2", "text": "Second.", "supersedes_id": None},
            {"id": "m3", "text": "Third.", "supersedes_id": None},
        )
        result = recall_memories(
            "Use the first and third constraints.",
            records,
            client=FakeClient(response(["m1", "m3"])),
        )
        self.assertEqual(result["selected_ids"], ["m1", "m3"])
        self.assertEqual([memory["id"] for memory in result["selected_memories"]], ["m1", "m3"])

    def test_empty_active_set_short_circuits_provider(self):
        client = FakeClient(error=AssertionError("provider must not be called"))
        result = recall_memories("Any request", [], client=client)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selected_ids"], [])
        self.assertEqual(result["selected_memories"], [])
        self.assertEqual(client.calls, [])

    def test_superseded_memory_is_excluded_from_provider_input(self):
        records = apply_memory_decision(
            [],
            {"action": "write", "memory": "Old preference.", "supersedes_id": None},
            new_id="m1",
        )
        records = apply_memory_decision(
            records,
            {"action": "update", "memory": "New preference.", "supersedes_id": "m1"},
            new_id="m2",
        )
        client = FakeClient(response(["m2"]))
        result = recall_memories("Use the preference.", records, client=client)
        body = json.loads(client.calls[0]["input"])
        self.assertEqual([memory["id"] for memory in body["active_memories"]], ["m2"])
        self.assertEqual(result["selected_ids"], ["m2"])
        self.assertEqual([memory["id"] for memory in active_memories(records)], ["m2"])

    def test_malformed_unknown_and_provider_errors_fail_closed(self):
        records = [{"id": "m1", "text": "Use concise reports.", "supersedes_id": None}]
        for client, error_type in (
            (FakeClient("not json"), "malformed_response"),
            (FakeClient(response(["missing"])), "malformed_response"),
            (FakeClient(error=RuntimeError("offline")), "provider_error"),
        ):
            result = recall_memories("Write a report.", records, client=client)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_type"], error_type)
            self.assertEqual(result["selected_ids"], [])
            self.assertEqual(result["selected_memories"], [])
            self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
