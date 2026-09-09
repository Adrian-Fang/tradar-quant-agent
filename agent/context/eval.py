"""Context Engineering Eval v1.1: behavior-contract and pair-level scoring."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json, load_prompt


CASES = load_json("eval/context_eval.json")
PAIR_EXPECTATIONS = {
    "momentum_evidence_state": "should_change",
    "high52_result_availability": "should_change",
    "current_truth_over_stale_state": "not_scored",
    "current_instruction_over_preference": "should_change",
    "distractor_robustness": "should_not_change",
    "comparison_context_sufficiency": "should_change",
}


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    pair_cases = [candidate for candidate in CASES if candidate["pair_id"] == case["pair_id"]]
    allowed_state_tags = sorted({
        tag
        for candidate in pair_cases
        for key in ("must_recognize", "must_not_recognize")
        for tag in candidate["behavior_contract"]["state"][key]
    })
    allowed_decisions = sorted({
        tag
        for candidate in pair_cases
        for key in ("acceptable", "must_not")
        for tag in candidate["behavior_contract"]["decision"][key]
    })
    allowed_action_tags = sorted({
        tag
        for candidate in pair_cases
        for key in ("must", "must_not", "optional")
        for tag in candidate["behavior_contract"]["action"][key]
    })
    return {
        "model": "",
        "instructions": load_prompt("prompts/context_eval.md"),
        "input": json.dumps(
            {
                "user_request": case["user_request"],
                "context": case["context"],
                "allowed_state_tags": allowed_state_tags,
                "allowed_decisions": allowed_decisions,
                "allowed_action_tags": allowed_action_tags,
            },
            ensure_ascii=False,
        ),
    }


class FixtureClient:
    """Deterministic harness response; no model call is made."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        contract = self.case["behavior_contract"]
        decision = contract["decision"]["acceptable"][0]
        actions = list(contract["action"]["must"])
        if decision == "evaluate_factor_first":
            actions.append("run_factor_evaluation")
        return {
            "output_text": json.dumps(
                {
                    "state": contract["state"]["must_recognize"],
                    "decision": decision,
                    "actions": list(dict.fromkeys(actions)),
                    "reason": "deterministic fixture response satisfies the behavior contract",
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


def _response_shape_error(parsed: dict[str, Any] | None) -> str | None:
    if parsed is None:
        return None
    expected = {"state", "decision", "actions", "reason"}
    if set(parsed) != expected:
        return "response must contain exactly state, decision, actions, and reason"
    if not isinstance(parsed["state"], list) or not all(isinstance(tag, str) for tag in parsed["state"]):
        return "response.state must be an array of strings"
    if not isinstance(parsed["decision"], str):
        return "response.decision must be a string"
    if not isinstance(parsed["actions"], list) or not all(isinstance(tag, str) for tag in parsed["actions"]):
        return "response.actions must be an array of strings"
    if not isinstance(parsed["reason"], str):
        return "response.reason must be a string"
    return None


def run_case(
    case: dict[str, Any], *, client: Any, model: str,
) -> dict[str, Any]:
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
    contract = case["behavior_contract"]
    state = set(parsed.get("state", [])) if isinstance(parsed, dict) else set()
    actions = set(parsed.get("actions", [])) if isinstance(parsed, dict) else set()
    decision = parsed.get("decision") if isinstance(parsed, dict) else None

    state_contract = contract["state"]
    decision_contract = contract["decision"]
    action_contract = contract["action"]
    missing_state = sorted(set(state_contract["must_recognize"]) - state)
    forbidden_state = sorted(set(state_contract["must_not_recognize"]) & state)
    decision_not_acceptable = decision not in set(decision_contract["acceptable"])
    forbidden_decision = decision in set(decision_contract["must_not"])
    missing_actions = sorted(set(action_contract["must"]) - actions)
    forbidden_actions = sorted(set(action_contract["must_not"]) & actions)

    if forbidden_state:
        state_status = "fail"
    elif missing_state:
        state_status = "partial"
    else:
        state_status = "pass"
    state_pass = state_status == "pass"
    decision_pass = not decision_not_acceptable and not forbidden_decision
    action_pass = not missing_actions and not forbidden_actions
    deterministic_failure = state_status == "fail" or not decision_pass or not action_pass
    review_reason = outcome["parse_error"] or outcome["structure_error"]
    if review_reason:
        status = "human_review"
        failure_type = "provider_error" if outcome["parse_error"] and outcome["parse_error"].startswith("provider_error:") else "malformed_response"
    elif deterministic_failure:
        status = "fail"
        failure_type = "behavior_contract_violation"
    else:
        status = "pass"
        failure_type = "none"

    actual_actions = sorted(actions)
    return {
        "case": case["id"],
        "pair_id": case["pair_id"],
        "group": case["group"],
        "repeat": repeat,
        "state": sorted(state),
        "state_status": state_status,
        "decision": decision or "<none>",
        "actions": actual_actions,
        "optional_action_tags": sorted(set(action_contract["optional"])),
        "state_interpretation_pass": state_pass,
        "decision_policy_pass": decision_pass,
        "action_compliance_pass": action_pass,
        "deterministic_pass": not deterministic_failure,
        "automated_pass": status == "pass",
        "status": status,
        "human_review_required": status == "human_review",
        "human_review_reason": review_reason or "",
        "failure_type": failure_type,
        "review_reason": review_reason or "",
        "parse_error": outcome["parse_error"],
        "structure_error": outcome["structure_error"],
        "missing_state": missing_state,
        "forbidden_state": forbidden_state,
        "decision_not_acceptable": decision_not_acceptable,
        "forbidden_decision": forbidden_decision,
        "missing_actions": missing_actions,
        "forbidden_actions": forbidden_actions,
        "reason": parsed.get("reason", "") if isinstance(parsed, dict) else "",
        "response_text": outcome["response_text"],
    }


def _pair_signature(row: dict[str, Any]) -> dict[str, Any]:
    optional_actions = set(row.get("optional_action_tags", []))
    return {
        "decision": row["decision"],
        "actions": sorted(set(row["actions"]) - optional_actions),
    }


def _pair_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for pair_id, expectation in PAIR_EXPECTATIONS.items():
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        result = {
            "pair_id": pair_id,
            "expectation": expectation,
            "cases": sorted({row["case"] for row in pair_rows}),
        }
        repeat_results = []
        for repeat in sorted({row["repeat"] for row in pair_rows}):
            repeat_rows = [row for row in pair_rows if row["repeat"] == repeat]
            signatures = sorted({
                json.dumps(_pair_signature(row), ensure_ascii=False, sort_keys=True)
                for row in repeat_rows
            })
            signature_values = [json.loads(signature) for signature in signatures]
            repeat_result = {
                "repeat": repeat,
                "signatures": signature_values,
                "distinct_signature_count": len(signature_values),
            }
            if expectation == "not_scored":
                repeat_result.update({"status": "not_scored", "pair_pass": None})
            elif any(row["human_review_required"] for row in repeat_rows):
                repeat_result.update({"status": "human_review", "pair_pass": None})
            else:
                changed = len(signature_values) > 1
                pair_pass = changed if expectation == "should_change" else not changed
                repeat_result.update({"status": "pass" if pair_pass else "fail", "pair_pass": pair_pass})
            repeat_results.append(repeat_result)
        result["repeat_results"] = repeat_results
        results.append(result)
    return results


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
            outcome = run_case(case, client=case_client, model=model)
            rows.append(score_case(case, outcome, repeat))
    return rows, {
        "status": "complete",
        "provider": provider,
        "cases": len(CASES),
        "repeats": repeats,
        "pair_results": _pair_results(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Engineering Eval v1.1")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Context Eval v1.1 skipped: {meta['reason']}")
        return
    for row in rows:
        print(
            f"{row['case']} | {row['status']} | "
            f"decision={row['decision']} | actions={row['actions']}"
        )
        if row["human_review_required"]:
            print(f"  review: {row['review_reason']}")
    print()
    print("Pair results by repeat:")
    for pair in meta["pair_results"]:
        for repeat in pair["repeat_results"]:
            print(
                f"{pair['pair_id']} | repeat={repeat['repeat']} | {repeat['status']} | "
                f"expectation={pair['expectation']} | "
                f"signatures={repeat['distinct_signature_count']}"
            )
    if args.verbose:
        print(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
