"""Deterministic eval runner for write-time memory decisions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json, load_prompt


CASES = load_json("eval/memory.json")
PROMPT = load_prompt("prompts/memory.md")
ACTIONS = {"write", "update", "ignore"}


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "",
        "instructions": PROMPT,
        "input": json.dumps({
            "message": case["message"],
            "existing_memories": case["existing_memories"],
        }, ensure_ascii=False),
    }


class FixtureClient:
    """Deterministic harness response; case oracle stays outside the prompt."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        expected = self.case["expected"]
        action = expected["action"]
        return {
            "output_text": json.dumps({
                "action": action,
                "memory": self.case.get("fixture_memory") if action != "ignore" else None,
                "supersedes_id": expected["supersedes_id"],
                "reason": "deterministic fixture decision",
            }, ensure_ascii=False),
        }


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def parse_memory_response(
    text: str,
    memory_ids: set[str],
) -> tuple[Any | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return parsed, "response JSON must be an object"
    if set(parsed) != {"action", "memory", "supersedes_id", "reason"}:
        return parsed, "response must contain exactly action, memory, supersedes_id, and reason"
    if not isinstance(parsed["action"], str) or parsed["action"] not in ACTIONS:
        return parsed, "response.action is invalid"
    if not isinstance(parsed["reason"], str):
        return parsed, "response.reason must be a string"
    if parsed["supersedes_id"] is not None and not isinstance(parsed["supersedes_id"], str):
        return parsed, "response.supersedes_id must be a string or null"

    action = parsed["action"]
    memory = parsed["memory"]
    supersedes_id = parsed["supersedes_id"]
    if action in {"write", "update"} and (not isinstance(memory, str) or not memory.strip()):
        return parsed, f"{action} requires a non-empty memory"
    if action == "write" and supersedes_id is not None:
        return parsed, "write must not supersede an existing memory"
    if action == "update":
        if supersedes_id not in memory_ids:
            return parsed, "update must cite an existing memory ID"
    if action == "ignore" and (memory is not None or supersedes_id is not None):
        return parsed, "ignore requires null memory and supersedes_id"
    return parsed, None


def run_case(case: dict[str, Any], *, client: Any, model: str) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    memory_ids = {item["id"] for item in case["existing_memories"]}
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
    parsed, validation_error = parse_memory_response(text, memory_ids)
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
    base = {
        "case": case["id"],
        "slice": case["slice"],
        "repeat": repeat,
        "expected_action": case["expected"]["action"],
        "actual_action": None,
        "actual_memory": None,
        "actual_supersedes_id": None,
        "action_correct": None,
        "supersedes_correct": None,
        "memory_content_correct": None,
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

    expected = case["expected"]
    actual = outcome["parsed"]
    action = actual["action"]
    memory = actual["memory"]
    supersedes_id = actual["supersedes_id"]
    action_correct = action == expected["action"]
    supersedes_correct = supersedes_id == expected["supersedes_id"]
    if expected["action"] == "ignore":
        memory_content_correct = memory is None and supersedes_id is None
    else:
        memory_text = memory.casefold() if isinstance(memory, str) else ""
        memory_content_correct = bool(memory_text) and all(
            any(alternative.casefold() in memory_text for alternative in concept_group)
            for concept_group in expected.get("required_tokens", [])
        ) and not any(
            any(alternative.casefold() in memory_text for alternative in concept_group)
            for concept_group in expected.get("forbidden_tokens", [])
        )
    case_pass = action_correct and supersedes_correct and memory_content_correct
    base.update({
        "actual_action": action,
        "actual_memory": memory,
        "actual_supersedes_id": supersedes_id,
        "action_correct": action_correct,
        "supersedes_correct": supersedes_correct,
        "memory_content_correct": memory_content_correct,
        "case_pass": case_pass,
        "failure_type": "none" if case_pass else "memory_decision_mismatch",
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
        "action_accuracy": average("action_correct"),
        "supersedes_accuracy": average("supersedes_correct"),
        "memory_content_accuracy": average("memory_content_correct"),
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
    parser = argparse.ArgumentParser(description="Write-time Memory Decision Eval")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Memory Eval skipped: {meta['reason']}")
        return

    print(f"Memory Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | expected | actual | pass")
    for row in rows:
        print(
            f"{row['case']} | {row['slice']} | {row['expected_action']} | "
            f"{row['actual_action'] or '-'} | {str(row['case_pass']).lower()}"
        )
    print("\nMetrics")
    print("slice | action | supersedes | memory_content | case_pass")
    for name, metrics in meta["slice_metrics"].items():
        value = lambda key: "-" if metrics[key] is None else f"{metrics[key]:.3f}"
        print(
            f"{name} | {value('action_accuracy')} | {value('supersedes_accuracy')} | "
            f"{value('memory_content_accuracy')} | {value('case_pass_rate')}"
        )
    overall = meta["metrics"]
    failure_rate = overall["eval_failure_rate"]
    print(
        f"\nOverall eval_failure_rate: "
        f"{'-' if failure_rate is None else f'{failure_rate:.3f}'}"
    )
    print(f"Failure breakdown: {meta['failure_breakdown'] or {}}")
    failures = [row for row in rows if row["failure_type"] != "none"]
    if failures:
        print("\nFailure details")
        print("case | repeat | failure_type | expected | actual | actual_memory | actual_supersedes_id | review_reason")
        for row in failures:
            print(
                f"{row['case']} | {row['repeat']} | {row['failure_type']} | "
                f"{row['expected_action']} | {row['actual_action'] or '-'} | "
                f"{row['actual_memory'] or '-'} | "
                f"{row['actual_supersedes_id'] or '-'} | {row['review_reason']}"
            )


if __name__ == "__main__":
    main()


__all__ = [
    "CASES",
    "FixtureClient",
    "parse_memory_response",
    "run_case",
    "run_eval",
    "score_case",
]
