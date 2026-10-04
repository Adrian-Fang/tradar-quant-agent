"""Fixture-only entry-preflight contract eval; token units are synthetic provider usage."""

from __future__ import annotations

import json
from typing import Any

from .core.resources import load_json
from .core.telemetry import RunTelemetry, TelemetryClient
from .preflight import TOOL_NAME, run_preflight


CASES = load_json("eval/preflight.json")


class FixtureClient:
    provider = "fixture"

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "output_text": "",
            "output": [{"type": "function_call", "name": TOOL_NAME,
                        "arguments": json.dumps(self.case["fixture"], ensure_ascii=False)}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }


def run_case(case: dict[str, Any], *, client: Any | None = None) -> dict[str, Any]:
    telemetry = RunTelemetry()
    result = run_preflight(
        case["user_request"], client=TelemetryClient(
            client if client is not None else FixtureClient(case), telemetry, stage="preflight"
        ), model="deepseek-flash", conversation_history=case.get("history"),
        capabilities=["inspect_universe", "evaluate_factor", "run_backtest", "run_research_experiment"],
    )
    parsed = result["result"] or {}
    expected = case["expected"]
    summary = telemetry.envelope()["summary"]
    route_correct = parsed.get("outcome") == expected["outcome"]
    answer_correct = all(value in parsed.get("answer", "") for value in expected["answer_contains"])
    return {
        "case": case["id"], "slice": case["slice"], "outcome": parsed.get("outcome"),
        "case_pass": result["status"] == "ok" and route_correct and answer_correct,
        "provider_calls": summary["calls"], "total_tokens": summary["total_tokens"],
        "latency_ms": summary["provider_latency_ms"], "error_type": result["error_type"],
    }


def run_eval() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [run_case(case) for case in CASES]
    return rows, {"cases": len(rows), "passed": sum(row["case_pass"] for row in rows),
                  "provider_calls": sum(row["provider_calls"] for row in rows)}


def main() -> None:
    rows, summary = run_eval()
    print("Preflight fixture eval (synthetic usage) | case | outcome | pass | calls | tokens | ms | error")
    for row in rows:
        print(f"{row['case']} | {row['outcome']} | {row['case_pass']} | {row['provider_calls']} | "
              f"{row['total_tokens']} | {row['latency_ms']} | {row['error_type'] or '-'}")
    print(f"Passed {summary['passed']}/{summary['cases']} | calls={summary['provider_calls']}")


if __name__ == "__main__":
    main()
