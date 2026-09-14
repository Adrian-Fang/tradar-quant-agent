"""Provider-agnostic runtime planner for explicit research plans."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.resources import load_prompt
from ..tools.runner import TOOL_SCHEMAS


PROMPT = load_prompt("prompts/planning.md")
ALLOWED_STATUSES = {"ready", "needs_input", "no_action"}
TOOL_NAMES = {schema["name"] for schema in TOOL_SCHEMAS}
TOOL_PARAMETERS = {
    schema["name"]: schema["parameters"] for schema in TOOL_SCHEMAS
}


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def parse_plan_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON must be an object"
    if set(parsed) != {"status", "steps", "reason"}:
        return None, "response must contain exactly status, steps, and reason"
    if not isinstance(parsed["status"], str) or parsed["status"] not in ALLOWED_STATUSES:
        return None, "response.status is invalid"
    if not isinstance(parsed["steps"], list):
        return None, "response.steps must be an array"
    if not isinstance(parsed["reason"], str):
        return None, "response.reason must be a string"
    for step in parsed["steps"]:
        if not isinstance(step, dict) or set(step) != {"name", "arguments"}:
            return None, "each step must contain exactly name and arguments"
        if not isinstance(step["name"], str) or step["name"] not in TOOL_NAMES:
            return None, "step.name is not an available tool"
        if not isinstance(step["arguments"], dict):
            return None, "step.arguments must be an object"
        parameters = TOOL_PARAMETERS[step["name"]]
        missing = sorted(set(parameters.get("required", [])) - set(step["arguments"]))
        if missing:
            return None, f"step.arguments is missing required fields: {missing}"
        if parameters.get("additionalProperties") is False:
            unknown = sorted(set(step["arguments"]) - set(parameters.get("properties", {})))
            if unknown:
                return None, f"step.arguments contains unknown fields: {unknown}"
    if parsed["status"] == "ready" and not parsed["steps"]:
        return None, "ready response must contain at least one step"
    if parsed["status"] != "ready" and parsed["steps"]:
        return None, "needs_input and no_action responses must have empty steps"
    return parsed, None


def plan_request(
    user_request: str,
    *,
    client: Any,
    model: str = "",
) -> dict[str, Any]:
    """Ask an injected provider for a plan without executing any step."""
    payload = {
        "model": model,
        "instructions": PROMPT,
        "input": json.dumps({
            "user_request": user_request,
            "tool_schemas": TOOL_SCHEMAS,
        }, ensure_ascii=False),
    }
    try:
        response = client.create(payload)
    except Exception as exc:
        return {
            "status": "error",
            "plan": None,
            "error_type": "provider_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    plan, parse_error = parse_plan_response(_response_text(response))
    if parse_error:
        return {
            "status": "error",
            "plan": None,
            "error_type": "malformed_response",
            "error": parse_error,
        }
    return {
        "status": "ok",
        "plan": plan,
        "error_type": None,
        "error": "",
    }


__all__ = ["parse_plan_response", "plan_request"]
