"""Deterministic eval runner for research planning decisions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json
from ..tools.calling import TOOL_SCHEMAS
from .planner import PROMPT, parse_plan_response


CASES = load_json("eval/planning.json")


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "",
        "instructions": PROMPT,
        "input": json.dumps({
            "user_request": case["request"],
            "tool_schemas": TOOL_SCHEMAS,
        }, ensure_ascii=False),
    }


class FixtureClient:
    """Deterministic plan response; case oracle stays outside the prompt."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        return {
            "output_text": json.dumps({
                "status": self.case["expected_status"],
                "steps": self.case["expected_steps"],
                "reason": "deterministic fixture plan",
            }, ensure_ascii=False),
        }


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def run_case(
    case: dict[str, Any],
    *,
    client: Any,
    model: str,
) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    try:
        response = client.create(payload)
    except Exception as exc:
        text = ""
        parsed = None
        parse_error = f"provider_error: {type(exc).__name__}: {exc}"
    else:
        text = _response_text(response)
        parsed, parse_error = parse_plan_response(text)
    return {
        "response_text": text,
        "parsed": parsed,
        "parse_error": parse_error,
    }


def score_case(
    case: dict[str, Any], outcome: dict[str, Any], repeat: int,
) -> dict[str, Any]:
    parsed = outcome["parsed"]
    review_reason = outcome["parse_error"] or ""
    if review_reason:
        return {
            "case": case["id"],
            "slice": case["slice"],
            "repeat": repeat,
            "expected_status": case["expected_status"],
            "expected_steps": case["expected_steps"],
            "actual_status": None,
            "status_correct": False,
            "required_step_recall": None,
            "step_precision": None,
            "unnecessary_step_rate": None,
            "order_correct": None,
            "argument_accuracy": None,
            "stop_correct": None,
            "case_pass": False,
            "eval_status": "human_review",
            "failure_type": "provider_error" if review_reason.startswith("provider_error:") else "malformed_response",
            "review_reason": review_reason,
            "actual_steps": [],
        }

    actual_status = parsed["status"]
    actual_steps = parsed["steps"]
    expected_status = case["expected_status"]
    expected_steps = case["expected_steps"]
    expected_names = [step["name"] for step in expected_steps]
    actual_names = [step["name"] for step in actual_steps]

    remaining_names = list(actual_names)
    matched_names = 0
    for name in expected_names:
        if name in remaining_names:
            remaining_names.remove(name)
            matched_names += 1
    required_step_recall = matched_names / len(expected_names) if expected_names else 1.0
    step_precision = matched_names / len(actual_names) if actual_names else (1.0 if not expected_names else 0.0)
    unnecessary_step_rate = 1.0 - step_precision
    order_correct = actual_names == expected_names

    argument_accuracy = (
        sum(expected_step == actual_step for expected_step, actual_step in zip(expected_steps, actual_steps))
        / len(expected_steps)
        if expected_steps else (1.0 if not actual_steps else 0.0)
    )
    status_correct = actual_status == expected_status
    stop_correct = (
        status_correct and not actual_steps
        if expected_status in {"needs_input", "no_action"}
        else None
    )
    case_pass = (
        status_correct
        and order_correct
        and required_step_recall == 1.0
        and step_precision == 1.0
        and argument_accuracy == 1.0
        and (expected_status == "ready" or not actual_steps)
    )
    return {
        "case": case["id"],
        "slice": case["slice"],
        "repeat": repeat,
        "expected_status": expected_status,
        "expected_steps": expected_steps,
        "actual_status": actual_status,
        "status_correct": status_correct,
        "required_step_recall": required_step_recall,
        "step_precision": step_precision,
        "unnecessary_step_rate": unnecessary_step_rate,
        "order_correct": order_correct,
        "argument_accuracy": argument_accuracy,
        "stop_correct": stop_correct,
        "case_pass": case_pass,
        "eval_status": "ok",
        "failure_type": "none" if case_pass else "plan_mismatch",
        "review_reason": "",
        "actual_steps": actual_steps,
    }


def _average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row["eval_status"] == "ok" and row[key] is not None]
    return sum(values) / len(values) if values else None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["eval_status"] == "ok"]
    return {
        "cases": len(rows),
        "eval_failures": len(rows) - len(valid),
        "status_correctness": _average(rows, "status_correct"),
        "required_step_recall": _average(rows, "required_step_recall"),
        "step_precision": _average(rows, "step_precision"),
        "unnecessary_step_rate": _average(rows, "unnecessary_step_rate"),
        "order_correctness": _average(rows, "order_correct"),
        "argument_accuracy": _average(rows, "argument_accuracy"),
        "stop_correctness": _average(rows, "stop_correct"),
        "case_pass_rate": sum(row["case_pass"] for row in valid) / len(valid) if valid else None,
    }


def run_eval(
    provider: str = "fixture", repeats: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if provider not in {"fixture", "openai", "deepseek"}:
        raise ValueError(f"unsupported provider: {provider}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        return [], {"status": "skipped", "reason": "DEEPSEEK_API_KEY is not set"}
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return [], {"status": "skipped", "reason": "OPENAI_API_KEY is not set"}

    model = (
        os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if provider == "deepseek" else os.getenv("OPENAI_MODEL", "gpt-5")
    )
    client = (
        DeepSeekChatClient() if provider == "deepseek"
        else OpenAIResponsesClient() if provider == "openai"
        else None
    )
    rows = []
    for case in CASES:
        for repeat in range(1, repeats + 1):
            case_client = FixtureClient(case) if provider == "fixture" else client
            rows.append(score_case(
                case,
                run_case(case, client=case_client, model=model),
                repeat,
            ))

    groups = {"overall": rows}
    for row in rows:
        groups.setdefault(row["slice"], []).append(row)
    return rows, {
        "status": "complete",
        "provider": provider,
        "cases": len(CASES),
        "repeats": repeats,
        "metrics": _metrics(rows),
        "slice_metrics": {name: _metrics(group) for name, group in groups.items()},
        "failure_breakdown": dict(Counter(
            row["failure_type"] for row in rows if row["failure_type"] != "none"
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Planning Eval")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Research Planning Eval skipped: {meta['reason']}")
        return

    print(f"Research Planning Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | expected | actual | pass")
    for row in rows:
        print(
            f"{row['case']} | {row['slice']} | {row['expected_status']} | "
            f"{row['actual_status'] or '-'} | {str(row['case_pass']).lower()}"
        )
    failures = [row for row in rows if row["eval_status"] != "ok" or not row["case_pass"]]
    if failures:
        print("\nFailure details")
        print("case | repeat | failure_type | expected_status | expected_steps | actual_status | actual_steps | review_reason")
        for row in failures:
            expected_steps = json.dumps(row["expected_steps"], ensure_ascii=False, separators=(",", ":"))
            actual_steps = json.dumps(row["actual_steps"], ensure_ascii=False, separators=(",", ":"))
            print(
                f"{row['case']} | {row['repeat']} | {row['failure_type']} | "
                f"{row['expected_status']} | {expected_steps} | "
                f"{row['actual_status'] or '-'} | {actual_steps} | {row['review_reason']}"
            )
    print("\nMetrics")
    print("slice | status | recall | precision | unnecessary | order | args | stop | case_pass")
    for name, metrics in meta["slice_metrics"].items():
        value = lambda key: "-" if metrics[key] is None else f"{metrics[key]:.3f}"
        print(
            f"{name} | {value('status_correctness')} | {value('required_step_recall')} | "
            f"{value('step_precision')} | {value('unnecessary_step_rate')} | "
            f"{value('order_correctness')} | {value('argument_accuracy')} | "
            f"{value('stop_correctness')} | {value('case_pass_rate')}"
        )
    if meta["failure_breakdown"]:
        print(f"Eval failures: {meta['failure_breakdown']}")


if __name__ == "__main__":
    main()
