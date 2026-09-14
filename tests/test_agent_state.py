from __future__ import annotations

import unittest

from agent.core.contracts import ResearchRun, ToolResult


def tool_result(tool_name: str, status: str = "success") -> ToolResult:
    return ToolResult(tool_name=tool_name, status=status)


class ResearchRunLifecycleTests(unittest.TestCase):
    def test_multiple_steps_keep_ordered_sequence(self):
        run = ResearchRun(run_id="run-1")

        run.add_step(tool_result("inspect_universe"))
        run.add_step(tool_result("evaluate_factor"))

        self.assertEqual([step["seq"] for step in run.steps], [1, 2])
        self.assertEqual(run.status, "running")

    def test_restore_then_append_continues_sequence(self):
        run = ResearchRun(run_id="run-1", user_request="research")
        run.add_step(tool_result("inspect_universe"))
        restored = ResearchRun.from_json(run.to_json())

        restored.add_step(tool_result("evaluate_factor"))

        self.assertEqual(restored.run_id, "run-1")
        self.assertEqual(restored.user_request, "research")
        self.assertEqual([step["seq"] for step in restored.steps], [1, 2])
        self.assertEqual(restored.status, "running")

    def test_complete_is_explicit_and_terminal(self):
        run = ResearchRun()

        run.complete()
        restored = ResearchRun.from_json(run.to_json())

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.final_status, "success")
        self.assertEqual(restored.status, "completed")
        self.assertEqual(restored.final_status, "success")

    def test_fail_is_explicit_and_terminal(self):
        run = ResearchRun()

        run.fail()

        self.assertEqual(run.status, "failed")
        self.assertEqual(run.final_status, "error")

    def test_legacy_final_status_derives_lifecycle(self):
        for final_status, status in (
            ("running", "running"),
            ("success", "completed"),
            ("partial", "completed"),
            ("error", "failed"),
        ):
            with self.subTest(final_status=final_status):
                restored = ResearchRun.from_dict({
                    "run_id": "legacy-run",
                    "final_status": final_status,
                })
                self.assertEqual(restored.status, status)
                self.assertEqual(restored.final_status, final_status)

    def test_inconsistent_lifecycle_and_final_status_is_rejected(self):
        for status, final_status in (("completed", "error"), ("failed", "success")):
            with self.subTest(status=status, final_status=final_status):
                with self.assertRaises(ValueError):
                    ResearchRun(status=status, final_status=final_status)

        with self.assertRaises(ValueError):
            ResearchRun.from_dict({"run_id": "legacy-run", "final_status": "unknown"})

    def test_terminal_run_rejects_new_steps(self):
        run = ResearchRun()
        run.complete()

        with self.assertRaises(RuntimeError):
            run.add_step(tool_result("evaluate_factor"))

    def test_terminal_transitions_are_rejected(self):
        run = ResearchRun()
        run.complete()

        with self.assertRaises(RuntimeError):
            run.complete()
        with self.assertRaises(RuntimeError):
            run.fail()

    def test_step_error_does_not_end_run(self):
        run = ResearchRun()

        run.add_step(tool_result("inspect_universe", status="error"))
        run.add_step(tool_result("evaluate_factor"))

        self.assertEqual(run.status, "running")
        self.assertEqual([step["seq"] for step in run.steps], [1, 2])


if __name__ == "__main__":
    unittest.main()
