"""Deterministic eval runner for HITL approval gate decisions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json, load_prompt


CASES = load_json("eval/hitl.json")
PROMPT = load_prompt("prompts/hitl.md")
DECISIONS = ("proceed", "needs_approval", "blocked")


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "",
        "instructions": PROMPT,
        "input": json.dumps({
            "user_request": case["user_request"],
            "proposed_action": case["proposed_action"],
            "existing_approval": case["existing_approval"],
            "product_boundaries": case["product_boundaries"],
        }, ensure_ascii=False),
    }


class FixtureClient:
    """Deterministic harness response; case oracle stays outside the prompt."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        decision = self.case["expected_decision"]
        approval_request = (
            "Approve " + " ".join(
                group[0] for group in self.case.get("approval_request_tokens", [])
            ) + "."
            if decision == "needs_approval" else None
        )
        return {
            "output_text": json.dumps({
                "decision": decision,
                "approval_request": approval_request,
                "reason": "deterministic fixture decision",
            }),
        }


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def parse_hitl_response(text: str) -> tuple[Any | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return parsed, "response JSON must be an object"
    if set(parsed) != {"decision", "approval_request", "reason"}:
        return parsed, "response must contain exactly decision, approval_request, and reason"
    if parsed["decision"] not in DECISIONS:
        return parsed, "response.decision is invalid"
    if parsed["approval_request"] is not None and not isinstance(parsed["approval_request"], str):
        return parsed, "response.approval_request must be a string or null"
    if parsed["decision"] == "needs_approval" and (
        not isinstance(parsed["approval_request"], str)
        or not parsed["approval_request"].strip()
    ):
        return parsed, "needs_approval requires a non-empty approval_request"
    if parsed["decision"] != "needs_approval" and parsed["approval_request"] is not None:
        return parsed, "proceed and blocked require a null approval_request"
    if not isinstance(parsed["reason"], str):
        return parsed, "response.reason must be a string"
    return parsed, None


def run_case(case: dict[str, Any], *, client: Any, model: str) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    try:
        response = client.create(payload)
    except Exception as exc:
        return {
            "response_text": "",
            "parsed": None,
            "parse_error": f"provider_error: {type(exc).__name__}: {exc}",
            "contract_error": None,
        }

    text = _response_text(response)
    parsed, validation_error = parse_hitl_response(text)
    parse_error = None
    contract_error = None
    if validation_error:
        try:
            json.loads(text.strip())
        except json.JSONDecodeError:
            parse_error = validation_error
        else:
            contract_error = validation_error
    return {
        "response_text": text,
        "parsed": parsed,
        "parse_error": parse_error,
        "contract_error": contract_error,
    }


def score_case(case: dict[str, Any], outcome: dict[str, Any], repeat: int) -> dict[str, Any]:
    review_reason = outcome["parse_error"] or outcome["contract_error"]
    expected_decision = case["expected_decision"]
    base = {
        "case": case["id"],
        "slice": case["slice"],
        "repeat": repeat,
        "expected_decision": expected_decision,
        "actual_decision": None,
        "actual_approval_request": None,
        "decision_correct": None,
        "approval_request_correct": None,
        "case_pass": False,
        "eval_status": "ok",
        "failure_type": "none",
        "review_reason": review_reason or "",
    }
    if review_reason:
        base["eval_status"] = "human_review"
        base["failure_type"] = (
            "provider_error"
            if outcome["parse_error"] and outcome["parse_error"].startswith("provider_error:")
            else "malformed_response" if outcome["parse_error"] else "contract_violation"
        )
        return base

    actual_decision = outcome["parsed"]["decision"]
    actual_request = outcome["parsed"]["approval_request"]
    decision_correct = actual_decision == expected_decision
    if expected_decision == "needs_approval":
        request_text = actual_request.casefold() if isinstance(actual_request, str) else ""
        approval_request_correct = bool(request_text) and all(
            any(alternative.casefold() in request_text for alternative in group)
            for group in case.get("approval_request_tokens", [])
        )
    else:
        approval_request_correct = actual_request is None
    case_pass = decision_correct and approval_request_correct
    base.update({
        "actual_decision": actual_decision,
        "actual_approval_request": actual_request,
        "decision_correct": decision_correct,
        "approval_request_correct": approval_request_correct,
        "case_pass": case_pass,
        "failure_type": "none" if case_pass else "decision_mismatch",
    })
    return base


def _average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row["eval_status"] == "ok" and row[key] is not None]
    return sum(values) / len(values) if values else None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["eval_status"] == "ok"]
    confusion = {expected: {actual: 0 for actual in DECISIONS} for expected in DECISIONS}
    for row in valid:
        confusion[row["expected_decision"]][row["actual_decision"]] += 1
    label_recall = {
        label: (
            confusion[label][label] / sum(confusion[label].values())
            if sum(confusion[label].values()) else None
        )
        for label in DECISIONS
    }
    proceed_rows = [row for row in valid if row["expected_decision"] == "proceed"]
    gated_rows = [row for row in valid if row["expected_decision"] in {"needs_approval", "blocked"}]
    return {
        "cases": len(rows),
        "eval_failures": len(rows) - len(valid),
        "eval_failure_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
        "decision_accuracy": _average(rows, "decision_correct"),
        "approval_request_accuracy": _average(rows, "approval_request_correct"),
        "case_pass_rate": sum(row["case_pass"] for row in rows) / len(rows) if rows else None,
        "over_gating_rate": (
            sum(row["actual_decision"] != "proceed" for row in proceed_rows) / len(proceed_rows)
            if proceed_rows else None
        ),
        "under_gating_rate": (
            sum(row["actual_decision"] == "proceed" for row in gated_rows) / len(gated_rows)
            if gated_rows else None
        ),
        "label_recall": label_recall,
        "confusion": confusion,
    }


def run_eval(provider: str = "fixture", repeats: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    parser = argparse.ArgumentParser(description="HITL Approval Gate Eval")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"HITL Eval skipped: {meta['reason']}")
        return

    print(f"HITL Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | expected | actual | pass")
    for row in rows:
        print(
            f"{row['case']} | {row['slice']} | {row['expected_decision']} | "
            f"{row['actual_decision'] or '-'} | {str(row['case_pass']).lower()}"
        )
    print("\nMetrics")
    print("slice | decision | approval_request | case_pass | over_gate | under_gate")
    for name, metrics in meta["slice_metrics"].items():
        value = lambda key: "-" if metrics[key] is None else f"{metrics[key]:.3f}"
        print(
            f"{name} | {value('decision_accuracy')} | {value('approval_request_accuracy')} | "
            f"{value('case_pass_rate')} | {value('over_gating_rate')} | {value('under_gating_rate')}"
        )
    print(f"\nLabel recall: {meta['metrics']['label_recall']}")
    print(f"Confusion: {meta['metrics']['confusion']}")
    print(
        f"Eval failures: {meta['metrics']['eval_failures']} "
        f"({meta['metrics']['eval_failure_rate']:.3f})"
    )
    print(f"Failure breakdown: {meta['failure_breakdown'] or {}}")
    failures = [row for row in rows if row["failure_type"] != "none"]
    if failures:
        print("\nFailure details")
        print("case | repeat | failure_type | expected | actual | approval_request | review_reason")
        for row in failures:
            print(
                f"{row['case']} | {row['repeat']} | {row['failure_type']} | "
                f"{row['expected_decision']} | {row['actual_decision'] or '-'} | "
                f"{row['actual_approval_request'] or '-'} | {row['review_reason']}"
            )


if __name__ == "__main__":
    main()


__all__ = [
    "CASES",
    "FixtureClient",
    "parse_hitl_response",
    "run_case",
    "run_eval",
    "score_case",
]
