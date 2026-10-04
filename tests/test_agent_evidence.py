"""Bounded serialization contract, independent of runtime stage plumbing."""

from datetime import date, timedelta
import json
import unittest

from agent.agent import tool_results_to_evidence
from agent.core.contracts import ToolResult


class AgentEvidenceTests(unittest.TestCase):
    def test_evidence_adapter_uses_only_successful_actual_outputs(self):
        results = [
            ToolResult(
                tool_name="inspect_universe",
                run_id="r",
                status="success",
                result={"value": 10, "date": "2026-08-31"},
            ),
            ToolResult.error("inspect_universe", {}, "failed", "no data", run_id="r"),
        ]
        self.assertEqual(tool_results_to_evidence(results), [{
            "id": "step-1-inspect_universe",
            "text": '{"date":"2026-08-31","value":10}',
        }])

    def test_large_inspect_output_is_bounded_and_stable(self):
        daily_counts = [{
            "date": (date(2026, 1, 1) + timedelta(days=index - 1)).isoformat(),
            "eligible": index,
        } for index in range(1, 101)]
        snapshots = [{
            "date": (date(2026, 1, 1) + timedelta(days=30 * (index - 1))).isoformat(),
            "membership": {
                "eligible": [f"S{symbol:04d}" for symbol in range(100)],
                "trading": [f"S{symbol:04d}" for symbol in range(100)],
                "buyable": [f"S{symbol:04d}" for symbol in range(100)],
                "sellable": [f"S{symbol:04d}" for symbol in range(100)],
                "price_limit_known": [f"S{symbol:04d}" for symbol in range(100)],
            },
        } for index in range(1, 21)]
        result = ToolResult(
            tool_name="inspect_universe",
            run_id="r",
            status="success",
            result={
                "summary": {"requested_start": "2026-01-01", "requested_end": "2026-12-31"},
                "daily_counts": daily_counts,
                "snapshots": snapshots,
            },
        )

        evidence = tool_results_to_evidence([result])
        repeat = tool_results_to_evidence([result])
        self.assertEqual(evidence, repeat)
        self.assertLess(len(evidence[0]["text"]), 16000)
        payload = json.loads(evidence[0]["text"])
        self.assertEqual(payload["daily_counts"]["omitted_count"], 76)
        self.assertEqual(payload["daily_counts"]["items"][0]["date"], "2026-01-01")
        self.assertEqual(payload["daily_counts"]["items"][-1]["date"], "2026-04-10")
        self.assertEqual(payload["snapshots"]["omitted_count"], 12)
        membership = payload["snapshots"]["items"][0]["membership"]
        self.assertEqual(membership["eligible"]["omitted_count"], 76)
        self.assertEqual(membership["eligible"]["items"][0], "S0000")
        self.assertEqual(membership["eligible"]["items"][-1], "S0099")


if __name__ == "__main__":
    unittest.main()
