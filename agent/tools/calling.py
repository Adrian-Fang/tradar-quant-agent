"""Minimal LLM tool-calling adapter for the Phase 1 research tools."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from typing import Any

from ..core.contracts import ResearchRun, ToolResult
from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient, ProviderError
from ..core.resources import load_prompt
from .experiment import run_research_experiment
from .research import evaluate_factor, inspect_universe, run_backtest


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
                "exclude_st": {"type": ["boolean", "null"], "description": "Apply canonical ST exclusion; null uses the default."},
                "min_turnover_rate": {"type": ["number", "null"], "description": "Canonical minimum turnover-rate filter; null uses the default."},
                "min_listed_days": {"type": ["integer", "null"], "minimum": 0, "description": "Canonical minimum listed trading days; null uses the default."},
                "snapshot_dates": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Optional dates for membership snapshots.",
                },
            },
            "required": ["start_date", "end_date", "exclude_st", "min_turnover_rate", "min_listed_days", "snapshot_dates"],
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
                "factor": {"type": "string", "description": "Canonical factor name or YAML path confined to data/factor_defs/."},
                "observe_start": {"type": "string", "description": "Evaluation start date, YYYY-MM-DD."},
                "observe_end": {"type": "string", "description": "Evaluation end date, YYYY-MM-DD."},
                "warmup_days": {"type": ["integer", "null"], "minimum": 0, "description": "Minimum calendar warmup before evaluation; null uses the default."},
                "ic_horizons": {
                    "type": ["array", "null"],
                    "items": {"type": "integer", "minimum": 1},
                    "description": "Forward-return horizons for IC and groups.",
                },
                "ic_method": {"type": ["string", "null"], "enum": ["pearson", "kendall", "spearman", None]},
                "n_groups": {"type": ["integer", "null"], "minimum": 2, "description": "Number of cross-sectional groups; null uses the default."},
                "return_clip": {"type": ["number", "null"], "minimum": 0, "description": "Symmetric forward-return clip; null uses the default."},
            },
            "required": ["factor", "observe_start", "observe_end", "warmup_days", "ic_horizons", "ic_method", "n_groups", "return_clip"],
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
            "under the repo's .runtime/ directory for weights, adjusted close, and adjusted open; optional masks "
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
                "buyable": {"type": ["string", "null"], "description": "Optional canonical buyable mask artifact path under .runtime/."},
                "sellable": {"type": ["string", "null"], "description": "Optional canonical sellable mask artifact path under .runtime/."},
                "benchmark_returns": {"type": ["string", "null"], "description": "Optional benchmark daily-return artifact path under .runtime/."},
                "buy_cost": {"type": ["number", "null"], "minimum": 0, "description": "Buy cost as decimal; null uses the canonical default."},
                "sell_cost": {"type": ["number", "null"], "minimum": 0, "description": "Sell cost as decimal; null uses the canonical default."},
                "slippage": {"type": ["number", "null"], "minimum": 0, "description": "Additional symmetric execution slippage as decimal; null uses the default."},
            },
            "required": ["target_weights", "price_panel", "open_panel", "buyable", "sellable", "benchmark_returns", "buy_cost", "sell_cost", "slippage"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_research_experiment",
        "description": (
            "Request one custom research experiment when the existing deterministic "
            "tools cannot answer the request. Supply a structured research spec, "
            "never Python source or internal artifact paths. Execution requires "
            "Linux user-namespace isolation and the configured canonical data directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "method": {"type": "string"},
                        "inputs": {"type": "object"},
                        "assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "outputs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "objective",
                        "method",
                        "inputs",
                        "assumptions",
                        "outputs",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": ["spec"],
            "additionalProperties": False,
        },
        # inputs is an intentionally open research/data specification, not a fixed object.
        "strict": False,
    },
)

TOOL_FUNCTIONS = {
    "inspect_universe": inspect_universe,
    "evaluate_factor": evaluate_factor,
    "run_backtest": run_backtest,
    "run_research_experiment": run_research_experiment,
}


def normalize_tool_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Model nulls request existing defaults; direct Python calls keep their semantics."""
    schema = next((item for item in TOOL_SCHEMAS if item["name"] == name), None)
    if schema is None or not schema["strict"]:
        return dict(arguments)
    properties = schema["parameters"]["properties"]
    return {key: value for key, value in arguments.items()
            if value is not None or "null" not in properties.get(key, {}).get("type", [])}


def _request_payload(user_request: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": load_prompt("prompts/tool_calling.md"),
        "input": user_request,
        "tools": [
            dict(schema)
            for schema in TOOL_SCHEMAS
            if schema["name"] != "run_research_experiment"
        ],
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
            **normalize_tool_arguments(selected_tool, call["arguments"]),
            run_id=run.run_id,
        )
        step = run.add_step(tool_result)
        step.update({
            "selected_tool": selected_tool,
            "model_args": call["arguments"],
        })
        if tool_result.status == "error":
            run.fail()
        else:
            run.complete(final_status=tool_result.status)
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
    except ProviderError as exc:
        error = ToolResult.error("tool_calling", base_args, exc.code, str(exc), run_id=run.run_id)
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
    run.fail()
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
    "run_research_experiment",
    "run_tool_calling",
]
