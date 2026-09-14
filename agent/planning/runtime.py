"""Minimal plan-once, execute-many runtime integration."""

from __future__ import annotations

from typing import Any

from ..core.contracts import ResearchRun
from ..tools.executor import execute_steps
from .planner import plan_request


def run_planned_request(
    user_request: str,
    *,
    client: Any,
    model: str = "",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Plan a request once, then execute the validated steps sequentially."""
    planned = plan_request(user_request, client=client, model=model)
    if planned["status"] == "error":
        return {
            "status": "error",
            "plan": None,
            "research_run": None,
            "error_type": planned["error_type"],
            "error": planned["error"],
        }

    plan = planned["plan"]
    if plan["status"] != "ready":
        return {
            "status": "ok",
            "plan": plan,
            "research_run": None,
            "error_type": None,
            "error": "",
        }

    run = (
        ResearchRun(user_request=user_request)
        if run_id is None
        else ResearchRun(run_id=run_id, user_request=user_request)
    )
    try:
        run = execute_steps(plan["steps"], run=run)
    except Exception as exc:
        return {
            "status": "error",
            "plan": plan,
            "research_run": run,
            "error_type": "orchestration_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "ok",
        "plan": plan,
        "research_run": run,
        "error_type": None,
        "error": "",
    }


__all__ = ["run_planned_request"]
