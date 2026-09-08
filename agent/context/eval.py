"""Context Engineering Eval v0: paired, rule-scored context decisions."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_jsonl, load_prompt


CONTEXT_CASES = load_jsonl("eval/context/cases.jsonl")


def _all_action_tags() -> tuple[str, ...]:
    tags = {
        tag
        for case in CONTEXT_CASES
        for key in ("required_actions", "forbidden_actions")
        for tag in case["scoring"][key]
    }
    return tuple(sorted(tags))


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "",
        "instructions": load_prompt("prompts/context.eval.txt"),
        "input": json.dumps(
            {
                "user_request": case["user_request"],
                "context": case["context"],
                "context_labels": case["critical_context"],
                "allowed_decisions": sorted({
                    item["expected_behavior"]["decision"] for item in CONTEXT_CASES
                }),
                "allowed_action_tags": _all_action_tags(),
            },
            ensure_ascii=False,
        ),
    }


class ContextFixtureClient:
    """Deterministic harness response; no model call is made."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        behavior = self.case["expected_behavior"]
        scoring = self.case["scoring"]
        return {
            "output_text": json.dumps(
                {
                    "decision": behavior["decision"],
                    "actions": scoring["required_actions"],
                    "used_context": self.case["critical_context"],
                    "reason": behavior["should"],
                },
                ensure_ascii=False,
            )
        }


def _response_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_text(response: Any) -> str:
    direct = _response_field(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    choices = _response_field(response, "choices", []) or []
    if choices:
        message = _response_field(choices[0], "message")
        content = _response_field(message, "content", "")
        if isinstance(content, str):
            return content
    for item in _response_field(response, "output", []) or []:
        content = _response_field(item, "content", []) or []
        for part in content:
            text = _response_field(part, "text", "")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _parse_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].removesuffix("```").strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None, "response is not valid JSON"
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON must be an object"
    return parsed, None


def run_context_case(case: dict[str, Any], *, client: Any, model: str) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    try:
        response = client.create(payload)
        text = _response_text(response)
        parsed, parse_error = _parse_response(text)
    except Exception as exc:
        text = ""
        parsed = None
        parse_error = f"provider_error: {type(exc).__name__}: {exc}"
    return {"response_text": text, "parsed": parsed, "parse_error": parse_error}


def score_context_case(case: dict[str, Any], outcome: dict[str, Any], repeat: int) -> dict[str, Any]:
    parsed = outcome["parsed"]
    behavior = case["expected_behavior"]
    scoring = case["scoring"]
    actions = set(parsed.get("actions", [])) if isinstance(parsed, dict) else set()
    used_context = set(parsed.get("used_context", [])) if isinstance(parsed, dict) else set()
    decision = parsed.get("decision") if isinstance(parsed, dict) else None
    missing_context = sorted(set(case["critical_context"]) - used_context)
    missing_actions = sorted(set(scoring["required_actions"]) - actions)
    forbidden_actions = sorted(set(scoring["forbidden_actions"]) & actions)
    decision_ok = decision == behavior["decision"]
    automated_pass = (
        outcome["parse_error"] is None
        and decision_ok
        and not missing_context
        and not missing_actions
        and not forbidden_actions
    )
    return {
        "case": case["id"],
        "pair_id": case["pair_id"],
        "group": case["group"],
        "repeat": repeat,
        "decision": decision or "<none>",
        "expected_decision": behavior["decision"],
        "decision_ok": decision_ok,
        "missing_context": missing_context,
        "missing_actions": missing_actions,
        "forbidden_actions": forbidden_actions,
        "automated_pass": automated_pass,
        "status": "pass" if automated_pass else "human_review",
        "human_review_required": not automated_pass,
        "human_review_reason": (
            "free-form reason is recorded but not automatically scored"
            if automated_pass else outcome["parse_error"] or "deterministic checks did not pass"
        ),
        "parse_error": outcome["parse_error"],
        "reason": parsed.get("reason", "") if isinstance(parsed, dict) else "",
        "response_text": outcome["response_text"],
    }


def run_context_eval(
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
    for case in CONTEXT_CASES:
        for repeat in range(1, repeats + 1):
            case_client = ContextFixtureClient(case) if provider == "fixture" else client
            outcome = run_context_case(case, client=case_client, model=model)
            rows.append(score_context_case(case, outcome, repeat))
    return rows, {
        "status": "complete",
        "provider": provider,
        "cases": len(CONTEXT_CASES),
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Engineering Eval v0")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    rows, meta = run_context_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Context Eval v0 skipped: {meta['reason']}")
        return
    for row in rows:
        print(
            f"{row['case']} | {row['status']} | "
            f"decision={row['decision']} | expected={row['expected_decision']}"
        )
        if row["human_review_required"]:
            print(f"  review: {row['parse_error'] or row['missing_actions'] or row['forbidden_actions'] or row['missing_context']}")
    if args.verbose:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
