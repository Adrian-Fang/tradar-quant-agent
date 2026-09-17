"""Deterministic eval runner for answer synthesis."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json
from .synthesizer import (
    build_synthesis_payload,
    parse_synthesis_response,
    response_text,
)


CASES = load_json("eval/answer_synthesis.json")


def _prompt(case: dict[str, Any]) -> dict[str, str]:
    return build_synthesis_payload(case["user_request"], case["evidence"])


class FixtureClient:
    """Deterministic harness response; expected values stay outside the prompt."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        expected = self.case["expected"]
        return {
            "output_text": json.dumps({
                "status": expected["status"],
                "answer": expected["fixture_answer"],
                "evidence_ids": expected["evidence_ids"],
            }, ensure_ascii=False),
        }


def run_case(
    case: dict[str, Any], *, client: Any, model: str,
) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    evidence_ids = {item["id"] for item in case["evidence"]}
    try:
        response = client.create(payload)
    except Exception as exc:
        return {
            "response_text": "",
            "parsed": None,
            "parse_error": f"provider_error: {type(exc).__name__}: {exc}",
            "contract_error": None,
        }

    text = response_text(response)
    parsed, validation_error = parse_synthesis_response(text, evidence_ids)
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


def _concept_groups_match(text: str, groups: list[list[str]]) -> bool:
    text = text.casefold()
    return all(
        any(alternative.casefold() in text for alternative in group)
        for group in groups
    )


def score_case(
    case: dict[str, Any], outcome: dict[str, Any], repeat: int,
) -> dict[str, Any]:
    review_reason = outcome["parse_error"] or outcome["contract_error"]
    expected = case["expected"]
    base = {
        "case": case["id"],
        "slice": case["slice"],
        "repeat": repeat,
        "expected_status": expected["status"],
        "actual_status": None,
        "expected_evidence_ids": expected["evidence_ids"],
        "actual_evidence_ids": None,
        "status_correct": None,
        "evidence_precision": None,
        "evidence_recall": None,
        "required_content_coverage": None,
        "forbidden_content_violations": None,
        "case_pass": False,
        "eval_status": "ok",
        "failure_type": "none",
        "review_reason": review_reason or "",
        "actual_answer": None,
    }
    if review_reason:
        base["eval_status"] = "human_review"
        base["failure_type"] = (
            "provider_error"
            if outcome["parse_error"] and outcome["parse_error"].startswith("provider_error:")
            else "malformed_response" if outcome["parse_error"] else "contract_violation"
        )
        return base

    actual = outcome["parsed"]
    expected_ids = set(expected["evidence_ids"])
    actual_ids = set(actual["evidence_ids"])
    overlap = len(expected_ids & actual_ids)
    precision = overlap / len(actual_ids) if actual_ids else (1.0 if not expected_ids else 0.0)
    recall = overlap / len(expected_ids) if expected_ids else (1.0 if not actual_ids else 0.0)
    coverage = _concept_groups_match(actual["answer"], expected.get("required_content", []))
    forbidden = [
        group for group in expected.get("forbidden_content", [])
        if any(alternative.casefold() in actual["answer"].casefold() for alternative in group)
    ]
    status_correct = actual["status"] == expected["status"]
    case_pass = (
        status_correct
        and precision == 1.0
        and recall == 1.0
        and coverage
        and not forbidden
    )
    base.update({
        "actual_status": actual["status"],
        "actual_evidence_ids": actual["evidence_ids"],
        "status_correct": status_correct,
        "evidence_precision": precision,
        "evidence_recall": recall,
        "required_content_coverage": float(coverage),
        "forbidden_content_violations": len(forbidden),
        "case_pass": case_pass,
        "failure_type": "none" if case_pass else "synthesis_mismatch",
        "actual_answer": actual["answer"],
    })
    return base


def _average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row["eval_status"] == "ok" and row[key] is not None]
    return sum(values) / len(values) if values else None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["eval_status"] == "ok"]
    return {
        "cases": len(rows),
        "eval_failures": len(rows) - len(valid),
        "status_accuracy": _average(rows, "status_correct"),
        "evidence_selection_precision": _average(rows, "evidence_precision"),
        "evidence_selection_recall": _average(rows, "evidence_recall"),
        "required_content_coverage": _average(rows, "required_content_coverage"),
        "forbidden_content_violation_rate": (
            sum(row["forbidden_content_violations"] > 0 for row in valid) / len(valid)
            if valid else None
        ),
        "case_pass_rate": (
            sum(row["case_pass"] for row in rows) / len(rows)
            if rows else None
        ),
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

    return rows, {
        "status": "complete",
        "provider": provider,
        "cases": len(CASES),
        "repeats": repeats,
        "metrics": _metrics(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer Synthesis Eval")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Answer Synthesis Eval skipped: {meta['reason']}")
        return

    print(f"Answer Synthesis Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | expected | actual | evidence | content | pass")
    for row in rows:
        evidence = row["actual_evidence_ids"] or "-"
        print(
            f"{row['case']} | {row['slice']} | {row['expected_status']} | "
            f"{row['actual_status'] or '-'} | {evidence} | "
            f"{row['required_content_coverage'] if row['required_content_coverage'] is not None else '-'} | "
            f"{str(row['case_pass']).lower()}"
        )
    print("\nMetrics")
    for name, value in meta["metrics"].items():
        print(f"{name}: {'-' if value is None else f'{value:.3f}' if isinstance(value, float) else value}")
    failures = [row for row in rows if row["eval_status"] != "ok" or not row["case_pass"]]
    if failures:
        print("\nFailure details")
        print("case | repeat | failure_type | expected | actual | review_reason")
        for row in failures:
            print(
                f"{row['case']} | {row['repeat']} | {row['failure_type']} | "
                f"{row['expected_status']} | {row['actual_status'] or '-'} | {row['review_reason']}"
            )


if __name__ == "__main__":
    main()


__all__ = ["CASES", "FixtureClient", "run_case", "run_eval", "score_case"]
