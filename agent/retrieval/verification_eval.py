"""Deterministic eval runner for query-to-research-record relevance verification."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json, load_prompt
from .loader import load_research_records


CASES = load_json("eval/relevance_verification.json")


def _prompt(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "",
        "instructions": load_prompt("prompts/relevance_verification.md"),
        "input": json.dumps(
            {"query": case["query"], "research_record": record},
            ensure_ascii=False,
        ),
    }


class FixtureClient:
    """Deterministic harness response; the oracle is not placed in the prompt."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        return {
            "output_text": json.dumps(
                {
                    "supported": self.case["expected_supported"],
                    "reason": "deterministic fixture classification",
                }
            )
        }


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def _parse_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON must be an object"
    return parsed, None


def _response_shape_error(parsed: dict[str, Any] | None) -> str | None:
    if parsed is None:
        return None
    if set(parsed) != {"supported", "reason"}:
        return "response must contain exactly supported and reason"
    if not isinstance(parsed["supported"], bool):
        return "response.supported must be a boolean"
    if not isinstance(parsed["reason"], str):
        return "response.reason must be a string"
    return None


def run_case(
    case: dict[str, Any],
    record: dict[str, Any],
    *,
    client: Any,
    model: str,
) -> dict[str, Any]:
    payload = _prompt(case, record)
    payload["model"] = model
    try:
        response = client.create(payload)
        text = _response_text(response)
        parsed, parse_error = _parse_response(text)
    except Exception as exc:
        text = ""
        parsed = None
        parse_error = f"provider_error: {type(exc).__name__}: {exc}"
    return {
        "response_text": text,
        "parsed": parsed,
        "parse_error": parse_error,
        "structure_error": _response_shape_error(parsed) if parse_error is None else None,
    }


def score_case(
    case: dict[str, Any], outcome: dict[str, Any], repeat: int,
) -> dict[str, Any]:
    parsed = outcome["parsed"]
    actual = parsed["supported"] if isinstance(parsed, dict) and outcome["structure_error"] is None else None
    review_reason = outcome["parse_error"] or outcome["structure_error"]
    if review_reason:
        status = "human_review"
        failure_type = (
            "provider_error"
            if outcome["parse_error"] and outcome["parse_error"].startswith("provider_error:")
            else "malformed_response"
        )
    else:
        status = "pass" if actual == case["expected_supported"] else "fail"
        failure_type = "none" if status == "pass" else "verification_mismatch"
    return {
        "case": case["id"],
        "slice": case.get("slice", "baseline"),
        "difficulty": case.get("difficulty", "normal"),
        "research_id": case["research_id"],
        "repeat": repeat,
        "expected_supported": case["expected_supported"],
        "actual_supported": actual,
        "status": status,
        "automated_pass": status == "pass",
        "human_review_required": status == "human_review",
        "failure_type": failure_type,
        "review_reason": review_reason or "",
        "reason": parsed.get("reason", "") if isinstance(parsed, dict) else "",
        "parse_error": outcome["parse_error"],
        "structure_error": outcome["structure_error"],
    }


def _rate(rows: list[dict[str, Any]], predicate) -> float | None:
    valid = [row for row in rows if not row["human_review_required"]]
    return sum(predicate(row) for row in valid) / len(valid) if valid else None


def _slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["expected_supported"]]
    negative = [row for row in rows if not row["expected_supported"]]
    hard_negative = [row for row in negative if row["difficulty"] == "hard_negative"]
    near_miss_negative = [row for row in negative if row["difficulty"] == "near_miss_negative"]
    return {
        "cases": len(rows),
        "human_review": sum(row["human_review_required"] for row in rows),
        "accuracy": _rate(rows, lambda row: row["actual_supported"] == row["expected_supported"]),
        "positive_acceptance": _rate(positive, lambda row: row["actual_supported"] is True),
        "negative_rejection": _rate(negative, lambda row: row["actual_supported"] is False),
        "hard_negative_rejection": _rate(hard_negative, lambda row: row["actual_supported"] is False),
        "near_miss_negative_rejection": _rate(near_miss_negative, lambda row: row["actual_supported"] is False),
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

    records = {record["research_id"]: record for record in load_research_records()}
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
        record = records[case["research_id"]]
        for repeat in range(1, repeats + 1):
            case_client = FixtureClient(case) if provider == "fixture" else client
            outcome = run_case(case, record, client=case_client, model=model)
            rows.append(score_case(case, outcome, repeat))

    groups = {"overall": rows}
    for row in rows:
        groups.setdefault(row["slice"], []).append(row)
    return rows, {
        "status": "complete",
        "provider": provider,
        "cases": len(CASES),
        "repeats": repeats,
        "slice_metrics": {
            name: _slice_metrics(group) for name, group in groups.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Relevance Verification Eval")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Relevance Verification Eval skipped: {meta['reason']}")
        return

    print(f"Relevance Verification Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | repeat | expected | actual | status")
    for row in rows:
        actual = "-" if row["actual_supported"] is None else str(row["actual_supported"]).lower()
        print(
            f"{row['case']} | {row['slice']} | {row['repeat']} | "
            f"{str(row['expected_supported']).lower()} | {actual} | {row['status']}"
        )
    print("\nSlice metrics")
    print(
        "slice | accuracy | positive_acceptance | negative_rejection | "
        "hard_negative_rejection | near_miss_negative_rejection | human_review"
    )
    for name, metrics in meta["slice_metrics"].items():
        def value(key: str) -> str:
            metric = metrics[key]
            return "-" if metric is None else f"{metric:.3f}"

        print(
            f"{name} | {value('accuracy')} | {value('positive_acceptance')} | "
            f"{value('negative_rejection')} | {value('hard_negative_rejection')} | "
            f"{value('near_miss_negative_rejection')} | {metrics['human_review']}"
        )


if __name__ == "__main__":
    main()
