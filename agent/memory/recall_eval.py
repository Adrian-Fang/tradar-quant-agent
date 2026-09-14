"""Deterministic eval runner for memory recall decisions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json
from .recall import RECALL_PROMPT, build_recall_payload, parse_recall_response, response_text


CASES = load_json("eval/memory_recall.json")
PROMPT = RECALL_PROMPT


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    return build_recall_payload(case["user_request"], case["active_memories"])


class FixtureClient:
    """Deterministic harness response; recall oracle stays outside the prompt."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        return {
            "output_text": json.dumps({
                "selected_ids": self.case["expected_selected_ids"],
                "reason": "deterministic fixture selection",
            }),
        }


def run_case(case: dict[str, Any], *, client: Any, model: str) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    memory_ids = [item["id"] for item in case["active_memories"]]
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
    parsed, validation_error = parse_recall_response(text, memory_ids)
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


def score_case(
    case: dict[str, Any], outcome: dict[str, Any], repeat: int,
) -> dict[str, Any]:
    review_reason = outcome["parse_error"] or outcome["contract_error"]
    expected = set(case["expected_selected_ids"])
    base = {
        "case": case["id"],
        "slice": case["slice"],
        "repeat": repeat,
        "expected_selected_ids": case["expected_selected_ids"],
        "actual_selected_ids": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "exact_match": None,
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

    actual = outcome["parsed"]["selected_ids"]
    actual_set = set(actual)
    correct = actual_set == expected
    if not expected:
        precision = recall = f1 = 1.0 if not actual else 0.0
    elif not actual:
        precision = recall = f1 = 0.0
    else:
        hits = len(expected & actual_set)
        precision = hits / len(actual_set)
        recall = hits / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    base.update({
        "actual_selected_ids": actual,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": correct,
        "case_pass": correct,
        "failure_type": "none" if correct else "selection_mismatch",
    })
    return base


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["eval_status"] == "ok"]

    def average(key: str) -> float | None:
        values = [row[key] for row in valid if row[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "cases": len(rows),
        "eval_failures": len(rows) - len(valid),
        "eval_failure_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
        "precision": average("precision"),
        "recall": average("recall"),
        "f1": average("f1"),
        "exact_match_rate": average("exact_match"),
        "case_pass_rate": sum(row["case_pass"] for row in rows) / len(rows) if rows else None,
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
    parser = argparse.ArgumentParser(description="Memory Recall Eval")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Memory Recall Eval skipped: {meta['reason']}")
        return

    print(f"Memory Recall Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | expected | actual | pass")
    for row in rows:
        print(
            f"{row['case']} | {row['slice']} | {row['expected_selected_ids']} | "
            f"{row['actual_selected_ids'] or '-'} | {str(row['case_pass']).lower()}"
        )
    print("\nMetrics")
    print("slice | precision | recall | f1 | exact_match | case_pass")
    for name, metrics in meta["slice_metrics"].items():
        value = lambda key: "-" if metrics[key] is None else f"{metrics[key]:.3f}"
        print(
            f"{name} | {value('precision')} | {value('recall')} | {value('f1')} | "
            f"{value('exact_match_rate')} | {value('case_pass_rate')}"
        )
    print(f"\nEval failures: {meta['metrics']['eval_failures']} ({meta['metrics']['eval_failure_rate']:.3f})")
    print(f"Failure breakdown: {meta['failure_breakdown'] or {}}")
    failures = [row for row in rows if row["failure_type"] != "none"]
    if failures:
        print("\nFailure details")
        print("case | repeat | failure_type | expected | actual | review_reason")
        for row in failures:
            print(
                f"{row['case']} | {row['repeat']} | {row['failure_type']} | "
                f"{row['expected_selected_ids']} | {row['actual_selected_ids'] or '-'} | "
                f"{row['review_reason']}"
            )


if __name__ == "__main__":
    main()


__all__ = [
    "CASES",
    "FixtureClient",
    "parse_recall_response",
    "run_case",
    "run_eval",
    "score_case",
]
