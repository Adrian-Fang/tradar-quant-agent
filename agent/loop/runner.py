"""Small bounded loop for deciding and executing one tool action at a time."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

from ..core.contracts import ResearchRun, ToolResult
from ..planning.planner import parse_plan_response
from ..tools.executor import execute_steps


def _validate_decision(decision: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(decision, Mapping):
        return None, "loop decision must be an object"
    if decision.get("status") == "error":
        if not set(decision) <= {"status", "error_type", "error", "error_stage"}:
            return None, "loop error decision has invalid fields"
        if not isinstance(decision["error_type"], str) or not isinstance(decision["error"], str):
            return None, "loop error decision fields are invalid"
        if "error_stage" in decision and not isinstance(decision["error_stage"], str):
            return None, "loop error decision error_stage must be a string"
        return dict(decision), None
    if set(decision) != {"status", "step", "reason"}:
        return None, "loop decision must contain exactly status, step, and reason"
    status = decision["status"]
    if status not in {"execute", "finish", "needs_input", "no_action", "blocked"}:
        return None, "loop decision.status is invalid"
    if not isinstance(decision["reason"], str):
        return None, "loop decision.reason must be a string"
    if status == "execute":
        step = decision["step"]
        if not isinstance(step, Mapping):
            return None, "execute decision.step must be an object"
        plan, error = parse_plan_response(
            json.dumps({
                "status": "ready",
                "steps": [dict(step)],
                "reason": decision["reason"],
            })
        )
        if error:
            return None, error
        return {"status": "execute", "step": plan["steps"][0], "reason": decision["reason"]}, None
    if decision["step"] is not None:
        return None, f"{status} decision.step must be null"
    return dict(decision), None


def _observation(
    user_request: str,
    run: ResearchRun | None,
    tool_results: list[ToolResult],
    iteration: int,
    provisional_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "user_request": user_request,
        "iteration": iteration,
        "steps": list(run.steps) if run is not None else [],
        "observations": [result.to_dict() for result in tool_results],
        "provisional_steps": list(provisional_steps),
    }


def run_loop(
    user_request: str,
    *,
    decide_next: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    initial_steps: list[dict[str, Any]] | None = None,
    run: ResearchRun | None = None,
    tool_results: list[ToolResult] | None = None,
    before_execute: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Run bounded one-action decisions against one append-only ResearchRun.

    ``decide_next`` receives prior steps and full ToolResult observations. If it
    is omitted, ``initial_steps`` are consumed as a compatibility path for the
    existing plan-once runtime.
    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if decide_next is None and not initial_steps:
        raise ValueError("initial_steps are required without decide_next")

    results = tool_results if tool_results is not None else []
    pending = list(initial_steps or [])
    iterations = 0
    first_decision = True

    while iterations < max_iterations:
        iterations += 1
        decided_by_loop = False
        if first_decision and pending:
            decision: Mapping[str, Any] = {
                "status": "execute",
                "step": pending.pop(0),
                "reason": "compatibility plan step",
            }
        elif decide_next is not None:
            decided_by_loop = True
            try:
                decision = decide_next(_observation(
                    user_request, run, results, iterations, pending,
                ))
            except Exception as exc:
                if run is not None and run.status == "running":
                    run.fail()
                return {
                    "status": "error",
                    "outcome": "error",
                    "research_run": run,
                    "tool_results": results,
                    "iterations": iterations,
                    "error_type": "provider_error",
                    "error_stage": "loop",
                "error": f"{type(exc).__name__}: {exc}",
            }
        elif pending:
            decision = {
                "status": "execute",
                "step": pending.pop(0),
                "reason": "compatibility plan step",
            }
        else:
            if run is not None and run.status == "running" and run.steps:
                run.complete(final_status=(
                    "partial"
                    if any(step["status"] == "partial" for step in run.steps)
                    else "success"
                ))
            return {
                "status": "ok",
                "outcome": "success",
                "research_run": run,
                "tool_results": results,
                "iterations": iterations - 1,
                "error_type": None,
                "error_stage": None,
                "error": "",
            }
        first_decision = False

        normalized, error = _validate_decision(decision)
        if error:
            if run is not None and run.status == "running":
                run.fail()
            return {
                "status": "error",
                "outcome": "error",
                "research_run": run,
                "tool_results": results,
                "iterations": iterations,
                "error_type": "malformed_response",
                "error_stage": "loop",
                "error": error,
            }
        if normalized["status"] == "error":
            if run is not None and run.status == "running":
                run.fail()
            return {
                "status": "error",
                "outcome": "error",
                "research_run": run,
                "tool_results": results,
                "iterations": iterations,
                "error_type": normalized["error_type"],
                "error_stage": normalized.get("error_stage", "loop"),
                "error": normalized["error"],
            }

        status = normalized["status"]
        if status != "execute":
            if run is not None and run.status == "running" and run.steps:
                run.complete(final_status=(
                    "partial"
                    if status != "finish" or any(
                        step["status"] == "partial" for step in run.steps
                    )
                    else "success"
                ))
            return {
                "status": "ok",
                "outcome": "success" if status == "finish" else status,
                "research_run": run,
                "tool_results": results,
                "iterations": iterations,
                "decision": normalized,
                "error_type": None,
                "error_stage": None,
                "error": "",
            }

        if run is None:
            run = ResearchRun(user_request=user_request)
        if decided_by_loop and pending:
            selected = normalized["step"]
            matching_index = next(
                (index for index, candidate in enumerate(pending) if candidate == selected),
                None,
            )
            pending = pending[matching_index + 1:] if matching_index is not None else pending[1:]
        if before_execute is not None:
            try:
                authorization = before_execute({
                    "user_request": user_request,
                    "step": normalized["step"],
                    "run": run,
                    "tool_results": results,
                })
            except Exception as exc:
                if run.status == "running":
                    run.fail()
                return {
                    "status": "error",
                    "outcome": "error",
                    "research_run": run,
                    "tool_results": results,
                    "iterations": iterations,
                    "error_type": "orchestration_error",
                    "error_stage": "hitl",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if not isinstance(authorization, Mapping):
                if run.status == "running":
                    run.fail()
                return {
                    "status": "error",
                    "outcome": "error",
                    "research_run": run,
                    "tool_results": results,
                    "iterations": iterations,
                    "error_type": "orchestration_error",
                    "error_stage": "hitl",
                    "error": "before_execute must return an object",
                }
            if authorization.get("status") == "error":
                if run.status == "running":
                    run.fail()
                return {
                    "status": "error",
                    "outcome": "error",
                    "research_run": run,
                    "tool_results": results,
                    "iterations": iterations,
                    "error_type": authorization.get("error_type", "orchestration_error"),
                    "error_stage": authorization.get("error_stage", "hitl"),
                    "error": authorization.get("error", "before_execute failed"),
                }
            if authorization.get("status") == "stop":
                outcome = authorization.get("outcome")
                if outcome not in {"needs_approval", "blocked"}:
                    if run.status == "running":
                        run.fail()
                    return {
                        "status": "error",
                        "outcome": "error",
                        "research_run": run,
                        "tool_results": results,
                        "iterations": iterations,
                        "error_type": "orchestration_error",
                        "error_stage": "hitl",
                        "error": "before_execute returned an invalid stop outcome",
                    }
                if run.status == "running" and run.steps:
                    run.complete(final_status="partial")
                return {
                    "status": "ok",
                    "outcome": outcome,
                    "research_run": run,
                    "tool_results": results,
                    "iterations": iterations,
                    "error_type": None,
                    "error_stage": None,
                    "error": "",
                }
            if authorization.get("status") != "proceed":
                if run.status == "running":
                    run.fail()
                return {
                    "status": "error",
                    "outcome": "error",
                    "research_run": run,
                    "tool_results": results,
                    "iterations": iterations,
                    "error_type": "orchestration_error",
                    "error_stage": "hitl",
                    "error": "before_execute returned an invalid status",
                }
        try:
            execute_steps(
                [normalized["step"]],
                run=run,
                tool_results=results,
                finalize=False,
            )
        except Exception as exc:
            if run.status == "running":
                run.fail()
            return {
                "status": "error",
                "outcome": "error",
                "research_run": run,
                "tool_results": results,
                "iterations": iterations,
                "error_type": "orchestration_error",
                "error_stage": "loop",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if run.status == "failed":
            return {
                "status": "ok",
                "outcome": "error",
                "research_run": run,
                "tool_results": results,
                "iterations": iterations,
                "decision": normalized,
                "error_type": "execution_error",
                "error_stage": "execution",
                "error": "tool execution failed",
            }
        if not pending and decide_next is None:
            run.complete(final_status=(
                "partial"
                if any(step["status"] == "partial" for step in run.steps)
                else "success"
            ))
            return {
                "status": "ok",
                "outcome": "success",
                "research_run": run,
                "tool_results": results,
                "iterations": iterations,
                "decision": normalized,
                "error_type": None,
                "error_stage": None,
                "error": "",
            }

    if run is not None and run.status == "running":
        run.fail()
    return {
        "status": "error",
        "outcome": "error",
        "research_run": run,
        "tool_results": results,
        "iterations": iterations,
        "error_type": "iteration_limit",
        "error_stage": "loop",
        "error": f"loop reached max_iterations={max_iterations}",
    }


__all__ = ["run_loop"]
