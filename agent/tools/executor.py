"""Deterministic sequential execution of explicitly ordered tool steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.contracts import ResearchRun, ToolResult
from .calling import TOOL_FUNCTIONS


def execute_steps(
    steps: list[dict[str, Any]],
    *,
    run: ResearchRun | None = None,
    user_request: str = "",
) -> ResearchRun:
    """Execute explicit tool steps in order and return the canonical trace."""
    if run is None:
        run = ResearchRun(user_request=user_request)
    elif run.status != "running":
        raise RuntimeError(f"cannot execute steps for ResearchRun from status: {run.status}")
    if not steps:
        raise ValueError("steps cannot be empty")

    saw_partial = any(step["status"] == "partial" for step in run.steps)
    for step in steps:
        step_name = "multistep"
        arguments: Any = {}
        try:
            step_name = step["name"]
            arguments = step.get("arguments", {})
            if step_name not in TOOL_FUNCTIONS:
                raise ValueError(f"unsupported tool: {step_name}")
            if not isinstance(arguments, Mapping):
                raise TypeError("step arguments must be a mapping")
            tool_result = TOOL_FUNCTIONS[step_name](
                **dict(arguments),
                run_id=run.run_id,
            )
            if not isinstance(tool_result, ToolResult):
                raise TypeError("tool must return a ToolResult")
        except Exception as exc:
            tool_name = step_name if isinstance(step_name, str) else "multistep"
            normalized_args = {
                "name": step_name,
                "arguments": arguments,
            }
            tool_result = ToolResult.error(
                tool_name,
                normalized_args,
                "step_execution_error",
                str(exc),
                run_id=run.run_id,
            )

        run.add_step(tool_result)
        if tool_result.status == "error":
            run.fail()
            break
        if tool_result.status == "partial":
            saw_partial = True
    else:
        run.complete(final_status="partial" if saw_partial else "success")

    return run


__all__ = ["execute_steps"]
