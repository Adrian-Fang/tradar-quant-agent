"""Minimal LLM tool-calling adapter for the Phase 1 research tools."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from typing import Any

from ..core.contracts import ResearchRun, ToolResult
from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_prompt
from .tools import evaluate_factor, inspect_universe, run_backtest


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "name": "inspect_universe",
        "description": (
            "Inspect Tradar's canonical A-share universe and execution masks. "
            "Use this for eligible/trading/buyable/sellable counts or snapshot "
            "membership. Do not use it to evaluate a factor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Inclusive YYYY-MM-DD date."},
                "end_date": {"type": "string", "description": "Inclusive YYYY-MM-DD date."},
                "exclude_st": {"type": "boolean", "description": "Apply canonical ST exclusion."},
                "min_turnover_rate": {"type": "number", "description": "Canonical minimum turnover-rate filter."},
                "min_listed_days": {"type": "integer", "minimum": 0, "description": "Canonical minimum listed trading days."},
                "snapshot_dates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional dates for membership snapshots.",
                },
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "evaluate_factor",
        "description": (
            "Evaluate one existing YAML factor with canonical research.factor_dsl "
            "and FactorAnalyzer. Use this for factor metadata, warmup-aware "
            "coverage, IC, group diagnostics, and yearly IC stability. Do not "
            "run a backtest or inspect the universe as a separate tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {"type": "string", "description": "Existing factor name or YAML path."},
                "observe_start": {"type": "string", "description": "Evaluation start date, YYYY-MM-DD."},
                "observe_end": {"type": "string", "description": "Evaluation end date, YYYY-MM-DD."},
                "warmup_days": {"type": "integer", "minimum": 0, "description": "Minimum calendar warmup before evaluation."},
                "ic_horizons": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": "Forward-return horizons for IC and groups.",
                },
                "ic_method": {"type": "string", "enum": ["pearson", "kendall", "spearman"]},
                "n_groups": {"type": "integer", "minimum": 2, "description": "Number of cross-sectional groups."},
                "return_clip": {"type": ["number", "null"], "minimum": 0, "description": "Optional symmetric forward-return clip."},
            },
            "required": ["factor", "observe_start", "observe_end"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_backtest",
        "description": (
            "Run already-formed target weights through Tradar's canonical "
            "T+1 vector backtest. Inputs are existing CSV/Parquet artifacts "
            "for weights, adjusted close, and adjusted open; optional masks "
            "and benchmark returns are accepted. Do not construct factors, "
            "select assets, or call another research tool. For factor "
            "predictive power, use evaluate_factor instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_weights": {"type": "string", "description": "Existing wide or date/symbol weight artifact path."},
                "price_panel": {"type": "string", "description": "Existing adjusted-close panel artifact path."},
                "open_panel": {"type": "string", "description": "Existing adjusted-open panel artifact path required for T+1."},
                "buyable": {"type": "string", "description": "Optional canonical buyable mask artifact path."},
                "sellable": {"type": "string", "description": "Optional canonical sellable mask artifact path."},
                "benchmark_returns": {"type": "string", "description": "Optional benchmark daily-return artifact path."},
                "buy_cost": {"type": ["number", "null"], "minimum": 0, "description": "Buy cost as decimal; null uses the canonical default."},
                "sell_cost": {"type": ["number", "null"], "minimum": 0, "description": "Sell cost as decimal; null uses the canonical default."},
                "slippage": {"type": "number", "minimum": 0, "description": "Additional symmetric execution slippage as decimal."},
            },
            "required": ["target_weights", "price_panel", "open_panel"],
            "additionalProperties": False,
        },
        "strict": True,
    },
)

TOOL_FUNCTIONS = {
    "inspect_universe": inspect_universe,
    "evaluate_factor": evaluate_factor,
    "run_backtest": run_backtest,
}


def _request_payload(user_request: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": load_prompt("prompts/tool_calling.md"),
        "input": user_request,
        "tools": [dict(schema) for schema in TOOL_SCHEMAS],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }


def _extract_function_calls(response: Any) -> list[dict[str, Any]]:
    def field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    calls = []
    for item in field(response, "output", []) or []:
        if field(item, "type") != "function_call":
            continue
        arguments = field(item, "arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, Mapping):
            raise ValueError("model function-call arguments must be a JSON object")
        calls.append({
            "name": field(item, "name"),
            "arguments": dict(arguments),
            "call_id": field(item, "call_id"),
        })
    return calls


def run_tool_calling(
    user_request: str,
    *,
    client: Any = None,
    model: str | None = None,
    run_id: str | None = None,
    provider: str = "openai",
) -> dict[str, Any]:
    """Select and execute one research tool, recording a ResearchRun trace."""
    run = ResearchRun(run_id=run_id or ResearchRun().run_id, user_request=user_request)
    if provider == "deepseek":
        model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    else:
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5")
    base_args = {"user_request": user_request, "model": model_name}

    try:
        if client is None:
            if provider == "deepseek":
                client = DeepSeekChatClient()
            elif provider == "openai":
                client = OpenAIResponsesClient()
            else:
                raise ValueError(f"unsupported provider: {provider}")
        response = client.create(_request_payload(user_request, model_name))
        calls = _extract_function_calls(response)
        if len(calls) != 1:
            raise ValueError(f"expected exactly one function call, received {len(calls)}")
        call = calls[0]
        selected_tool = call["name"]
        if selected_tool not in TOOL_FUNCTIONS:
            raise ValueError(f"unsupported model tool: {selected_tool}")
        tool_result = TOOL_FUNCTIONS[selected_tool](
            **call["arguments"],
            run_id=run.run_id,
        )
        step = run.add_step(tool_result)
        step.update({
            "selected_tool": selected_tool,
            "model_args": call["arguments"],
        })
        run.final_status = tool_result.status
        return {
            "selected_tool": selected_tool,
            "model_args": call["arguments"],
            "tool_result": tool_result,
            "research_run": run,
            "provider": type(client).__name__,
        }
    except json.JSONDecodeError as exc:
        error = ToolResult.error(
            "tool_calling",
            base_args,
            "invalid_model_tool_call",
            f"model returned invalid JSON arguments: {exc}",
            run_id=run.run_id,
        )
    except RuntimeError as exc:
        code = "llm_client_unavailable" if "API_KEY" in str(exc) else "llm_execution_error"
        error = ToolResult.error("tool_calling", base_args, code, str(exc), run_id=run.run_id)
    except (ValueError, TypeError) as exc:
        error = ToolResult.error(
            "tool_calling",
            base_args,
            "invalid_model_tool_call",
            str(exc),
            run_id=run.run_id,
        )
    except Exception as exc:
        error = ToolResult.error(
            "tool_calling",
            base_args,
            "llm_execution_error",
            f"provider execution failed: {type(exc).__name__}: {exc}",
            run_id=run.run_id,
        )
    run.add_step(error, result_summary={"selected_tool": None})
    run.final_status = "error"
    return {
        "selected_tool": None,
        "model_args": None,
        "tool_result": error,
        "research_run": run,
        "provider": (
            type(client).__name__ if client is not None
            else ("DeepSeekChatClient" if provider == "deepseek" else "OpenAIResponsesClient")
        ),
    }


__all__ = [
    "DeepSeekChatClient",
    "OpenAIResponsesClient",
    "TOOL_FUNCTIONS",
    "TOOL_SCHEMAS",
    "run_tool_calling",
]
