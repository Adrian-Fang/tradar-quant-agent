"""Deterministic whole-system evaluation over observed trace envelopes."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from .core.resources import load_json


CASES = load_json("eval/whole_system.json")
FAILURE_STAGES = {
    "context",
    "retrieval",
    "planning",
    "tool",
    "execution",
    "state",
    "grounding",
    "hitl",
    "orchestration",
    "outcome",
}
OUTCOME_STATUSES = {"success", "error", "abstain", "needs_approval", "blocked"}
REQUIRED_OBSERVED_FIELDS = {
    "context",
    "steps",
    "retrieval",
    "research_run",
    "grounding",
    "hitl",
    "outcome",
    "failure",
}
GROUNDING_LABELS = {"supported", "unsupported", "contradicted", "unverifiable"}


def _validate_observed(observed: Any) -> None:
    if not isinstance(observed, dict) or set(observed) != REQUIRED_OBSERVED_FIELDS:
        raise ValueError("observed envelope has invalid fields")

    context = observed["context"]
    if not isinstance(context, dict) or not isinstance(context.get("selected_ids"), list):
        raise ValueError("observed.context.selected_ids must be a list")
    if any(not isinstance(value, str) for value in context["selected_ids"]):
        raise ValueError("observed context ids must be strings")

    if not isinstance(observed["steps"], list):
        raise ValueError("observed.steps must be a list")
    for step in observed["steps"]:
        if (
            not isinstance(step, dict)
            or not {"name", "arguments", "status"}.issubset(step)
            or not isinstance(step["name"], str)
            or not isinstance(step["arguments"], dict)
            or step["status"] not in {"success", "partial", "error"}
        ):
            raise ValueError("observed step has invalid shape")

    retrieval = observed["retrieval"]
    if (
        not isinstance(retrieval, dict)
        or not isinstance(retrieval.get("status"), str)
        or not isinstance(retrieval.get("research_ids"), list)
    ):
        raise ValueError("observed.retrieval has invalid shape")
    if any(not isinstance(value, str) for value in retrieval["research_ids"]):
        raise ValueError("observed research ids must be strings")

    run = observed["research_run"]
    if (
        not isinstance(run, dict)
        or run.get("status") not in {"running", "completed", "failed"}
        or run.get("final_status") not in {"running", "success", "partial", "error"}
    ):
        raise ValueError("observed.research_run has invalid lifecycle")

    grounding = observed["grounding"]
    if (
        not isinstance(grounding, dict)
        or not isinstance(grounding.get("fully_grounded"), bool)
        or not isinstance(grounding.get("labels"), list)
        or any(label not in GROUNDING_LABELS for label in grounding["labels"])
    ):
        raise ValueError("observed.grounding has invalid shape")

    if not isinstance(observed["hitl"], dict) or not isinstance(observed["hitl"].get("decision"), str):
        raise ValueError("observed.hitl has invalid shape")
    if not isinstance(observed["outcome"], dict) or observed["outcome"].get("status") not in OUTCOME_STATUSES:
        raise ValueError("observed.outcome has invalid status")

    failure = observed["failure"]
    if failure is not None and (
        not isinstance(failure, dict)
        or failure.get("stage") not in FAILURE_STAGES
        or not isinstance(failure.get("type"), str)
    ):
        raise ValueError("observed.failure has invalid shape")


def _trajectory(expected_steps: list[dict[str, Any]], actual_steps: list[dict[str, Any]]) -> dict[str, Any]:
    used = set()
    matched_indexes = []
    for expected in expected_steps:
        match = next(
            (
                index
                for index, actual in enumerate(actual_steps)
                if index not in used
                and actual["name"] == expected["name"]
                and actual["arguments"] == expected["arguments"]
            ),
            None,
        )
        if match is not None:
            used.add(match)
            matched_indexes.append(match)

    required_count = len(expected_steps)
    actual_count = len(actual_steps)
    matched_count = len(matched_indexes)
    recall = matched_count / required_count if required_count else 1.0
    precision = matched_count / actual_count if actual_count else (1.0 if not required_count else 0.0)
    order = float(
        matched_count == required_count
        and matched_indexes == sorted(matched_indexes)
    )
    return {
        "required_step_recall": recall,
        "precision": precision,
        "unnecessary_step_rate": 1.0 - precision,
        "order_correctness": order,
    }


def _infer_failure(case: dict[str, Any], observed: dict[str, Any], checks: dict[str, bool]) -> tuple[str | None, str | None]:
    if observed["failure"] is not None:
        return observed["failure"]["stage"], observed["failure"]["type"]
    if all(checks.values()):
        return None, None
    if not checks["context"]:
        return "context", "context_mismatch"
    if not checks["retrieval"]:
        return "retrieval", "retrieval_mismatch"
    if any(step["status"] == "error" for step in observed["steps"]):
        return "execution", "tool_result_error"

    expected_steps = case["expected"].get("required_steps", [])
    actual_steps = observed["steps"]
    same_name_wrong_args = any(
        actual["name"] == expected["name"]
        and actual["arguments"] != expected["arguments"]
        for actual in actual_steps
        for expected in expected_steps
    )
    if same_name_wrong_args:
        return "tool", "wrong_arguments"
    if not checks["trajectory"]:
        return "planning", "trajectory_mismatch"
    if not checks["state"]:
        return "state", "lifecycle_mismatch"
    if not checks["grounding"]:
        return "grounding", "grounding_mismatch"
    if not checks["hitl"]:
        return "hitl", "decision_mismatch"
    if not checks["outcome"]:
        return "outcome", "outcome_mismatch"
    return "orchestration", "unclassified_failure"


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    return case["observed"]


def score_case(case: dict[str, Any], observed: dict[str, Any], repeat: int = 1) -> dict[str, Any]:
    base = {
        "case": case["id"],
        "slice": case.get("slice", "baseline"),
        "repeat": repeat,
        "eval_status": "ok",
        "eval_failure_type": None,
        "review_reason": "",
        "expected_outcome": case["expected"]["outcome"],
        "actual_outcome": None,
        "expected_failure_stage": case["expected"].get("failure_stage"),
        "expected_failure_type": case["expected"].get("failure_type"),
        "failure_stage": None,
        "failure_type": None,
        "outcome_pass": False,
        "case_pass": False,
        "trajectory": None,
        "grounding_pass": None,
        "hitl_correctness": None,
        "failure_attribution_correct": None,
    }
    try:
        _validate_observed(observed)
    except (TypeError, ValueError) as exc:
        base.update({
            "eval_status": "human_review",
            "eval_failure_type": "contract_violation",
            "review_reason": str(exc),
            "failure_stage": "orchestration",
            "failure_type": "contract_violation",
        })
        return base

    expected = case["expected"]
    actual_outcome = observed["outcome"]["status"]
    base["actual_outcome"] = actual_outcome
    trajectory = _trajectory(expected.get("required_steps", []), observed["steps"])
    base["trajectory"] = trajectory

    checks = {
        "context": observed["context"]["selected_ids"] == expected.get("context_ids", []),
        "retrieval": (
            observed["retrieval"]["status"] == expected.get("retrieval_status", "not_used")
            and observed["retrieval"]["research_ids"] == expected.get("retrieval_ids", [])
        ),
        "trajectory": (
            trajectory["required_step_recall"] == 1.0
            and trajectory["precision"] == 1.0
            and trajectory["order_correctness"] == 1.0
        ),
        "state": (
            observed["research_run"]["status"] == expected.get("run_status", "running")
            and observed["research_run"]["final_status"] == expected.get("final_status", "running")
        ),
        "grounding": observed["grounding"]["fully_grounded"] == expected.get("grounding", False),
        "hitl": observed["hitl"]["decision"] == expected.get("hitl_decision", "not_required"),
        "outcome": actual_outcome == expected["outcome"],
    }
    base["outcome_pass"] = checks["outcome"]
    base["grounding_pass"] = float(checks["grounding"])
    base["hitl_correctness"] = float(checks["hitl"])
    base["failure_stage"], base["failure_type"] = _infer_failure(case, observed, checks)

    expected_failure = expected.get("failure_stage") is not None
    if expected_failure:
        base["failure_attribution_correct"] = float(
            base["failure_stage"] == expected["failure_stage"]
            and base["failure_type"] == expected["failure_type"]
        )
        base["case_pass"] = bool(checks["outcome"] and base["failure_attribution_correct"])
    else:
        base["case_pass"] = all(checks.values())
    return base


def _average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if row["eval_status"] != "ok"]
    attribution_rows = [row for row in rows if row["failure_attribution_correct"] is not None]
    trajectory_rows = [row for row in rows if row["trajectory"] is not None]
    failure_breakdown = Counter(
        row["eval_failure_type"] or row["failure_type"]
        for row in failures
    )
    failure_stages = Counter(
        row["failure_stage"]
        for row in rows
        if row["failure_stage"] is not None
    )
    trajectory_metrics = {
        name: (
            sum(row["trajectory"][name] for row in trajectory_rows)
            / len(trajectory_rows)
            if trajectory_rows else None
        )
        for name in (
            "required_step_recall",
            "precision",
            "unnecessary_step_rate",
            "order_correctness",
        )
    }
    return {
        "cases": len(rows),
        "case_pass_rate": _average(rows, "case_pass"),
        "outcome_pass_rate": _average(rows, "outcome_pass"),
        "trajectory_required_step_recall": trajectory_metrics["required_step_recall"],
        "trajectory_precision": trajectory_metrics["precision"],
        "unnecessary_step_rate": trajectory_metrics["unnecessary_step_rate"],
        "order_correctness": trajectory_metrics["order_correctness"],
        "grounding_pass": _average(rows, "grounding_pass"),
        "hitl_correctness": _average(rows, "hitl_correctness"),
        "failure_attribution_accuracy": _average(attribution_rows, "failure_attribution_correct"),
        "eval_failures": len(failures),
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
        "failure_stage_breakdown": dict(sorted(failure_stages.items())),
    }


def run_eval(provider: str = "fixture", repeats: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if provider != "fixture":
        raise ValueError("whole-system eval currently supports fixture only")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    rows = []
    for case in CASES:
        for repeat in range(1, repeats + 1):
            rows.append(score_case(case, run_case(case), repeat))
    return rows, {
        "status": "complete",
        "provider": provider,
        "cases": len(CASES),
        "repeats": repeats,
        "metrics": _metrics(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-system deterministic eval")
    parser.add_argument("--provider", choices=("fixture",), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows, meta = run_eval(args.provider, args.repeats)
    print(f"Whole-system Eval | {meta['provider']} | {meta['cases']} cases x {meta['repeats']}")
    print("case | slice | outcome | pass | failure")
    for row in rows:
        failure = row["failure_stage"] or "-"
        if row["failure_type"]:
            failure += f"/{row['failure_type']}"
        print(f"{row['case']} | {row['slice']} | {row['actual_outcome'] or '-'} | {str(row['case_pass']).lower()} | {failure}")

    print("\nMetrics")
    for name, value in meta["metrics"].items():
        if isinstance(value, dict):
            print(f"{name}: {value}")
        elif isinstance(value, float):
            print(f"{name}: {value:.3f}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()


__all__ = ["CASES", "run_case", "run_eval", "score_case"]
