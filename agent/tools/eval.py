"""Deterministic benchmark runner for the Phase 1 research tools."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import expand_tokens, load_json
from .runner import run_tool_calling


EVAL_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "temp" / "agent_eval"
EVAL_ARTIFACTS = {
    "weights": EVAL_ARTIFACT_DIR / "weights.csv",
    "close": EVAL_ARTIFACT_DIR / "close.csv",
    "open": EVAL_ARTIFACT_DIR / "open.csv",
}

_ARTIFACT_TOKEN = "${EVAL_ARTIFACT_DIR}"
CASES = tuple(
    expand_tokens(case, {_ARTIFACT_TOKEN: str(EVAL_ARTIFACT_DIR)})
    for case in load_json("eval/tool_calling.json")
)


def _prepare_eval_artifacts() -> dict[str, Path]:
    """Write the small deterministic CSV fixture used by the local benchmark."""
    EVAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2026-01-05", periods=4, freq="D", name="date")
    frames = {
        "weights": pd.DataFrame({"000001": [1.0, 1.0, 0.0, 0.0]}, index=dates),
        "close": pd.DataFrame({"000001": [100.0, 110.0, 120.0, 120.0]}, index=dates),
        "open": pd.DataFrame({"000001": [100.0, 105.0, 115.0, 120.0]}, index=dates),
    }
    for name, frame in frames.items():
        frame.to_csv(EVAL_ARTIFACTS[name])
    return EVAL_ARTIFACTS


class FixtureClient:
    """Deterministic model-call fixture; execution still uses real agent tools."""

    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "output": [{
                "type": "function_call",
                "name": self.tool_name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            }],
        }


def score_case(case: dict[str, Any], outcome: dict[str, Any], repeat: int) -> dict[str, Any]:
    selected = outcome["selected_tool"]
    model_args = outcome["model_args"] or {}
    expected_args = case.get("expected_args", {})

    selection_ok = selected == case["expected_tool"]
    missing_args = [key for key in expected_args if key not in model_args]
    incorrect_args = {
        key: {"expected": expected_args[key], "actual": model_args[key]}
        for key in expected_args
        if key in model_args and model_args[key] != expected_args[key]
    }
    correct_expected_args = sum(
        1
        for key, expected_value in expected_args.items()
        if key in model_args and model_args[key] == expected_value
    )
    argument_accuracy = correct_expected_args / len(expected_args) if expected_args else 1.0

    extra_args = [key for key in model_args if key not in expected_args]
    forbidden_args = set(case.get("forbidden_args", []))
    forbidden_args_present = [key for key in model_args if key in forbidden_args]
    agent_decision_pass = (
        selection_ok
        and not missing_args
        and not incorrect_args
        and not forbidden_args_present
    )

    tool_result = outcome["tool_result"]
    execution_ok = tool_result.status == "success"
    error_code = tool_result.errors[0]["code"] if tool_result.errors else ""
    e2e_pass = agent_decision_pass and execution_ok

    if not selection_ok:
        failure_type = "selection_error"
    elif missing_args:
        failure_type = "missing_argument"
    elif incorrect_args:
        failure_type = "incorrect_argument"
    elif forbidden_args_present:
        failure_type = "forbidden_argument"
    elif not execution_ok:
        failure_type = "missing_data" if error_code == "missing_data" else "tool_execution_error"
    else:
        failure_type = "none"

    return {
        "case": case["id"],
        "repeat": repeat,
        "severity": case.get("severity", "medium"),
        "selected": selected or "<none>",
        "selection_ok": selection_ok,
        "argument_accuracy": argument_accuracy,
        "missing_args": missing_args,
        "incorrect_args": incorrect_args,
        "extra_args": extra_args,
        "forbidden_args": forbidden_args_present,
        "agent_decision_pass": agent_decision_pass,
        "execution_ok": execution_ok,
        "status": tool_result.status,
        "error_code": error_code,
        "e2e_pass": e2e_pass,
        "failure_type": failure_type,
        "model_args": model_args,
    }


def _skipped_metrics(provider: str, repeats: int, reason: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "scope": "agent_eval" if provider != "fixture" else "test_harness",
        "status": "skipped",
        "reason": reason,
        "cases": len(CASES),
        "repeats": repeats,
        "attempts": 0,
        "tool_selection_accuracy": None,
        "argument_accuracy": None,
        "execution_success": None,
        "case_pass_rate": None,
        "inconsistent_or_failed": [],
    }


def run_eval(provider: str = "deepseek", repeats: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if provider not in {"fixture", "openai", "deepseek"}:
        raise ValueError(f"unsupported provider: {provider}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        return [], _skipped_metrics(provider, repeats, "DEEPSEEK_API_KEY is not set")
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return [], _skipped_metrics(provider, repeats, "OPENAI_API_KEY is not set")

    _prepare_eval_artifacts()
    client = (
        DeepSeekChatClient() if provider == "deepseek"
        else OpenAIResponsesClient() if provider == "openai"
        else None
    )
    rows = []
    for case in CASES:
        for repeat in range(1, repeats + 1):
            case_client = (
                FixtureClient(case["expected_tool"], case["expected_args"])
                if provider == "fixture" else client
            )
            outcome = run_tool_calling(
                case["request"],
                client=case_client,
                model="deepseek-v4-flash" if provider == "deepseek" else None,
                provider=provider if provider != "fixture" else "openai",
            )
            rows.append(score_case(case, outcome, repeat))

    case_summaries = []
    for case in CASES:
        attempts = [row for row in rows if row["case"] == case["id"]]
        selections = sorted({row["selected"] for row in attempts})
        case_summaries.append({
            "case": case["id"],
            "severity": case.get("severity", "medium"),
            "attempts": len(attempts),
            "selected_tools": selections,
            "selection_consistent": len(selections) == 1,
            "selection_accuracy": sum(row["selection_ok"] for row in attempts) / len(attempts),
            "argument_accuracy": sum(row["argument_accuracy"] for row in attempts) / len(attempts),
            "agent_decision_pass_rate": sum(row["agent_decision_pass"] for row in attempts) / len(attempts),
            "execution_success": sum(row["execution_ok"] for row in attempts) / len(attempts),
            "e2e_pass_rate": sum(row["e2e_pass"] for row in attempts) / len(attempts),
            "agent_pass": all(row["agent_decision_pass"] for row in attempts),
            "e2e_pass": all(row["e2e_pass"] for row in attempts),
            "failure_types": sorted({
                row["failure_type"] for row in attempts if row["failure_type"] != "none"
            }),
        })

    failure_breakdown = Counter(
        row["failure_type"] for row in rows if row["failure_type"] != "none"
    )
    high_critical_failures = [
        {
            "case": row["case"],
            "repeat": row["repeat"],
            "severity": row["severity"],
            "failure_type": row["failure_type"],
        }
        for row in rows
        if row["severity"] in {"high", "critical"} and row["failure_type"] != "none"
    ]
    metrics = {
        "provider": provider,
        "scope": "test_harness" if provider == "fixture" else "agent_eval",
        "status": "complete",
        "cases": len(CASES),
        "repeats": repeats,
        "attempts": len(rows),
        "tool_selection_accuracy": sum(row["selection_ok"] for row in rows) / len(rows),
        "argument_accuracy": sum(row["argument_accuracy"] for row in rows) / len(rows),
        "agent_decision_pass_rate": sum(row["agent_decision_pass"] for row in rows) / len(rows),
        "execution_success": sum(row["execution_ok"] for row in rows) / len(rows),
        "e2e_pass_rate": sum(row["e2e_pass"] for row in rows) / len(rows),
        "case_agent_pass_rate": sum(summary["agent_pass"] for summary in case_summaries) / len(case_summaries),
        "case_e2e_pass_rate": sum(summary["e2e_pass"] for summary in case_summaries) / len(case_summaries),
        "case_summaries": case_summaries,
        "failure_breakdown": dict(failure_breakdown),
        "high_critical_failures": high_critical_failures,
        "failed_attempts": [row for row in rows if not row["e2e_pass"]],
    }
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Agent tool-calling benchmark")
    parser.add_argument(
        "--provider", default="deepseek", choices=("deepseek", "openai", "fixture"),
        help="deepseek/openai are Agent eval; fixture is test harness only",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", help="print detailed case and attempt JSON")
    args = parser.parse_args()
    rows, metrics = run_eval(args.provider, repeats=args.repeats)
    print(f"Agent Eval — {args.provider} | {metrics['cases']} cases × {args.repeats}")
    if metrics["status"] != "complete":
        print(f"Status: {metrics['status']}")
        print(f"Reason: {metrics.get('reason', '')}")
        return

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    print()
    print(f"Tool Selection:  {pct(metrics['tool_selection_accuracy'])}")
    print(f"Arguments:       {pct(metrics['argument_accuracy'])}")
    print(f"Agent Decision:  {pct(metrics['agent_decision_pass_rate'])}")
    print(f"Execution:       {pct(metrics['execution_success'])}")
    print(f"End-to-End:      {pct(metrics['e2e_pass_rate'])}")
    print()
    print(f"Case Pass:       Agent {pct(metrics['case_agent_pass_rate'])} | E2E {pct(metrics['case_e2e_pass_rate'])}")
    print()
    if metrics["failure_breakdown"]:
        print("Failures:")
        for failure_type, count in sorted(metrics["failure_breakdown"].items()):
            print(f"- {failure_type}: {count}")
    else:
        print("Failures: none")
    if metrics["high_critical_failures"]:
        print()
        print("High/Critical failures:")
        seen = set()
        for failure in metrics["high_critical_failures"]:
            key = (failure["case"], failure["severity"], failure["failure_type"])
            if key in seen:
                continue
            seen.add(key)
            print(f"- {failure['case']}: {failure['failure_type']} ({failure['severity']})")
    if args.verbose:
        print()
        print("Detailed metrics:")
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        print()
        print("Attempts:")
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
