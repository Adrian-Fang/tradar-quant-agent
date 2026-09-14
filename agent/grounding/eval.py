"""Deterministic eval runner for final-answer grounding."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
from ..core.resources import load_json, load_prompt
from .verifier import parse_grounding_response


CASES = load_json("eval/grounding.json")
PROMPT = load_prompt("prompts/grounding.md")
LABEL_ORDER = ("supported", "unsupported", "contradicted", "unverifiable")


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "",
        "instructions": PROMPT,
        "input": json.dumps({
            "answer": case["answer"],
            "evidence": case["evidence"],
        }, ensure_ascii=False),
    }


class FixtureClient:
    """Deterministic harness response; expected labels stay outside the prompt."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        return {
            "output_text": json.dumps({
                "answer": self.case["answer"],
                "claims": [
                    {
                        "claim": self.case["answer"],
                        "evidence_ids": expected["evidence_ids"],
                        "grounding": expected["label"],
                    }
                    for expected in self.case["expected_claims"]
                ],
            }, ensure_ascii=False),
        }


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        text = response.get("output_text", "")
    else:
        text = getattr(response, "output_text", "")
    return text if isinstance(text, str) else ""


def run_case(case: dict[str, Any], *, client: Any, model: str) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    evidence_ids = {item["id"] for item in case["evidence"]}
    try:
        response = client.create(payload)
        text = _response_text(response)
        parsed, validation_error = parse_grounding_response(text, evidence_ids)
    except Exception as exc:
        text = ""
        parsed = None
        parse_error = f"provider_error: {type(exc).__name__}: {exc}"
        structure_error = None
    else:
        parse_error = validation_error if parsed is None else None
        structure_error = validation_error if parsed is not None else None
    return {
        "response_text": text,
        "parsed": parsed,
        "parse_error": parse_error,
        "structure_error": structure_error,
    }


def score_case(
    case: dict[str, Any], outcome: dict[str, Any], repeat: int,
) -> dict[str, Any]:
    review_reason = outcome["parse_error"] or outcome["structure_error"]
    base = {
        "case": case["id"],
        "slice": case["slice"],
        "repeat": repeat,
        "expected_grounded": case["expected_grounded"],
        "actual_grounded": None,
        "claim_results": [],
        "claim_label_accuracy": None,
        "evidence_attribution_accuracy": None,
        "required_token_check": None,
        "answer_groundedness_correct": None,
        "case_pass": False,
        "eval_status": "ok",
        "failure_type": "none",
        "review_reason": review_reason or "",
        "actual_claims": [],
    }
    if review_reason:
        base["eval_status"] = "human_review"
        base["failure_type"] = (
            "provider_error"
            if outcome["parse_error"] and outcome["parse_error"].startswith("provider_error:")
            else "malformed_response" if outcome["parse_error"] else "contract_violation"
        )
        return base

    actual_claims = outcome["parsed"]["claims"]
    expected_claims = case["expected_claims"]
    claim_results = []
    matched_indexes = set()
    for expected in expected_claims:
        expected_ids = set(expected["evidence_ids"])
        matched = [
            (index, actual)
            for index, actual in enumerate(actual_claims)
            if expected_ids & set(actual["evidence_ids"])
        ]
        matched_indexes.update(index for index, _ in matched)
        actual_labels = {actual["grounding"] for _, actual in matched}
        actual_text = " ".join(actual["claim"] for _, actual in matched)
        required_tokens = expected.get("required_tokens", [])
        claim_results.append({
            "expected_label": expected["label"],
            "predicted_label": (
                next(iter(actual_labels)) if len(actual_labels) == 1
                else "mixed" if actual_labels else "missing"
            ),
            "label_correct": actual_labels == {expected["label"]},
            "attribution_correct": bool(matched) and expected_ids <= {
                evidence_id
                for _, actual in matched
                for evidence_id in actual["evidence_ids"]
            },
            "required_token_check": bool(matched) and all(
                token in actual_text for token in required_tokens
            ),
        })

    no_unmatched_claims = matched_indexes == set(range(len(actual_claims)))
    label_correct = all(item["label_correct"] for item in claim_results) and no_unmatched_claims
    attribution_correct = all(item["attribution_correct"] for item in claim_results) and no_unmatched_claims
    required_token_check = all(item["required_token_check"] for item in claim_results) and no_unmatched_claims
    actual_grounded = bool(actual_claims) and no_unmatched_claims and all(
        claim["grounding"] == "supported" for claim in actual_claims
    )
    answer_grounded_correct = actual_grounded == case["expected_grounded"]
    case_pass = label_correct and attribution_correct and required_token_check and answer_grounded_correct
    base.update({
        "actual_grounded": actual_grounded,
        "claim_results": claim_results,
        "claim_label_accuracy": sum(item["label_correct"] for item in claim_results) / len(expected_claims) if expected_claims else 1.0,
        "evidence_attribution_accuracy": sum(item["attribution_correct"] for item in claim_results) / len(expected_claims) if expected_claims else 1.0,
        "required_token_check": sum(item["required_token_check"] for item in claim_results) / len(expected_claims) if expected_claims else 1.0,
        "answer_groundedness_correct": answer_grounded_correct,
        "case_pass": case_pass,
        "failure_type": "none" if case_pass else "grounding_mismatch",
        "actual_claims": actual_claims,
    })
    return base


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["eval_status"] == "ok"]
    claims = [claim for row in valid for claim in row["claim_results"]]
    actual_claims = [claim for row in valid for claim in row["actual_claims"]]
    label_confusion = {label: {} for label in LABEL_ORDER}
    for claim in claims:
        expected = claim["expected_label"]
        predicted = claim["predicted_label"]
        label_confusion[expected][predicted] = label_confusion[expected].get(predicted, 0) + 1
    per_label_recall = {
        label: (
            sum(
                claim["expected_label"] == label and claim["predicted_label"] == label
                for claim in claims
            ) / sum(claim["expected_label"] == label for claim in claims)
            if any(claim["expected_label"] == label for claim in claims)
            else None
        )
        for label in LABEL_ORDER
    }
    return {
        "cases": len(rows),
        "eval_failures": len(rows) - len(valid),
        "claim_label_accuracy": (
            sum(claim["label_correct"] for claim in claims) / len(claims)
            if claims else None
        ),
        "unsupported_claim_rate": (
            sum(claim["grounding"] == "unsupported" for claim in actual_claims) / len(actual_claims)
            if actual_claims else None
        ),
        "contradiction_rate": (
            sum(claim["grounding"] == "contradicted" for claim in actual_claims) / len(actual_claims)
            if actual_claims else None
        ),
        "evidence_attribution_accuracy": (
            sum(claim["attribution_correct"] for claim in claims) / len(claims)
            if claims else None
        ),
        "answer_groundedness_accuracy": (
            sum(row["answer_groundedness_correct"] for row in valid) / len(valid)
            if valid else None
        ),
        "case_pass_rate": (
            sum(row["case_pass"] for row in valid) / len(valid)
            if valid else None
        ),
        "label_confusion": {
            label: values for label, values in label_confusion.items() if values
        },
        "per_label_recall": per_label_recall,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-answer Grounding Eval")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Grounding Eval skipped: {meta['reason']}")
        return

    print(f"Grounding Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | expected | actual | status")
    for row in rows:
        labels = [claim["grounding"] for claim in row["actual_claims"]]
        expected = [claim["expected_label"] for claim in row["claim_results"]]
        print(
            f"{row['case']} | {row['slice']} | {expected or '-'} | "
            f"{labels or '-'} | {'human_review' if row['eval_status'] != 'ok' else str(row['case_pass']).lower()}"
        )
    print("\nSlice metrics")
    print(
        "slice | claim_label_accuracy | unsupported_claim_rate | contradiction_rate | "
        "evidence_attribution_accuracy | answer_groundedness_accuracy"
    )
    for name, metrics in meta["slice_metrics"].items():
        def value(key: str) -> str:
            metric = metrics[key]
            return "-" if metric is None else f"{metric:.3f}"

        print(
            f"{name} | {value('claim_label_accuracy')} | "
            f"{value('unsupported_claim_rate')} | {value('contradiction_rate')} | "
            f"{value('evidence_attribution_accuracy')} | "
            f"{value('answer_groundedness_accuracy')}"
        )

        print("  label recall:")
        for label in LABEL_ORDER:
            recall = metrics["per_label_recall"][label]
            print(f"    {label}: {'-' if recall is None else f'{recall:.3f}'}")
        print("  confusion (expected -> actual):")
        for expected_label, counts in metrics["label_confusion"].items():
            print(f"    {expected_label}: {counts}")


if __name__ == "__main__":
    main()
