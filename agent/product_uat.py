"""Real-task UAT over the existing runtime; no model judge or runtime overrides."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from .core.resources import expand_tokens, load_json_value
from .grounding.verifier import parse_grounding_response
from .main import run_request
from .tools.experiment import (
    EXPERIMENT_PROVENANCE_FIELDS, validate_experiment_provenance,
    validate_experiment_result, validate_experiment_spec,
)


TOOLS = {"inspect_universe", "evaluate_factor", "run_backtest", "run_research_experiment"}
ROUTES = TOOLS | {"knowledge_only", "needs_input", "safety_blocked"}
ARTIFACTS = {"target_weights", "price_panel", "open_panel"}


def _usefulness(value: Any) -> None:
    if value is not None and (type(value) is not int or value not in (0, 1, 2)):
        raise ValueError("usefulness_0_2 must be a manual integer 0–2 or null")


def validate_cases(cases: Any) -> None:
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("cases must be a nonempty array")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not {"id", "request", "expected_route", "expected"} <= set(case) or set(case) - {"id", "request", "expected_route", "expected", "required_artifacts", "usefulness_0_2"}:
            raise ValueError("invalid UAT case fields")
        identity = case["id"]
        if not isinstance(identity, str) or not re.fullmatch(r"UAT-\d{2}", identity) or identity in seen:
            raise ValueError("case IDs must be unique UAT-NN strings")
        seen.add(identity)
        if not isinstance(case["request"], str) or not case["request"].strip() or not isinstance(case["expected_route"], str) or case["expected_route"] not in ROUTES:
            raise ValueError(f"{identity}: invalid request/expected_route")
        expected = case["expected"]
        if not isinstance(expected, dict) or not {"outcome", "records", "review_notes"} <= set(expected) or set(expected) - {"outcome", "records", "review_notes", "arguments", "answer_patterns", "record_patterns", "spec_terms", "spec_patterns"}:
            raise ValueError(f"{identity}: invalid expected fields")
        outcome = "needs_input" if case["expected_route"] == "needs_input" else "blocked" if case["expected_route"] == "safety_blocked" else "success"
        if expected["outcome"] != outcome or not isinstance(expected["review_notes"], str) or not expected["review_notes"].strip():
            raise ValueError(f"{identity}: invalid outcome/review_notes")
        records = expected["records"]
        if not isinstance(records, dict) or any(not isinstance(key, str) or not re.fullmatch(r"RR-\d{3}", key) or not isinstance(value, str) or value not in {"validated", "rejected", "inconclusive"} for key, value in records.items()):
            raise ValueError(f"{identity}: invalid records")
        patterns = expected.get("record_patterns", {})
        if not isinstance(patterns, dict) or set(patterns) - set(records):
            raise ValueError(f"{identity}: invalid record_patterns")
        for values in [expected.get(key, []) for key in ("answer_patterns", "spec_terms", "spec_patterns")] + list(patterns.values()):
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{identity}: patterns/terms must be string arrays")
            for pattern in values:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"{identity}: invalid pattern: {exc}") from exc
        if not isinstance(expected.get("arguments", {}), dict):
            raise ValueError(f"{identity}: arguments must be an object")
        artifacts = case.get("required_artifacts", [])
        if not isinstance(artifacts, list) or any(not isinstance(value, str) for value in artifacts) or (artifacts and (case["expected_route"] != "run_backtest" or set(artifacts) != ARTIFACTS or len(artifacts) != 3)):
            raise ValueError(f"{identity}: invalid required_artifacts")
        if case["expected_route"] in TOOLS - {"run_research_experiment"} and not expected.get("arguments"):
            raise ValueError(f"{identity}: fixed tool route needs expected arguments")
        if case["expected_route"] == "run_research_experiment" and not expected.get("spec_terms"):
            raise ValueError(f"{identity}: experiment needs fresh scope terms")
        if case["expected_route"] == "knowledge_only" and (not records or set(patterns) != set(records) or not all(patterns.values())):
            raise ValueError(f"{identity}: knowledge route needs record-specific acceptance patterns")
        if case["expected_route"] == "run_backtest" and not artifacts:
            raise ValueError(f"{identity}: backtest needs runtime artifact bindings")
        if artifacts and any(expected.get("arguments", {}).get(key) != f"${{{key}}}" for key in artifacts):
            raise ValueError(f"{identity}: artifact paths must be runtime placeholders")
        _usefulness(case.get("usefulness_0_2"))


def load_cases() -> tuple[dict[str, Any], ...]:
    dataset = load_json_value("eval/product_uat.json")
    if not isinstance(dataset, dict) or set(dataset) != {"schema_version", "cases"} or dataset["schema_version"] != "product-uat-v1":
        raise ValueError("unsupported product UAT dataset")
    validate_cases(dataset["cases"])
    return tuple(dataset["cases"])


CASES = load_cases()


def _patterns(text: str, patterns: list[str]) -> bool:
    return all(re.search(pattern, text, re.IGNORECASE) is not None for pattern in patterns)


def _run_dict(result: dict[str, Any]) -> dict[str, Any] | None:
    run = result.get("research_run")
    return run.to_dict() if hasattr(run, "to_dict") else run


def _has_number(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_number(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (isinstance(value, int) or math.isfinite(value))


def _task_checks(case: dict[str, Any], result: dict[str, Any], artifacts: dict[str, str]) -> dict[str, bool]:
    expected, route = case["expected"], case["expected_route"]
    observed, answer = result["observed"], result.get("answer") or ""
    evidence = {item["id"]: json.loads(item["text"]) for item in result.get("evidence") or []}
    claims = (result.get("grounding") or {}).get("claims", [])
    checks = {"answer": bool(answer) and _patterns(answer, expected.get("answer_patterns", []))}
    records = expected["records"]
    retrieval = observed.get("retrieval") or {}
    checks["context_records"] = set(records) <= set(retrieval.get("research_ids", [])) & set((observed.get("context") or {}).get("selected_ids", []))
    if route != "knowledge_only" and records:
        retrieved = {record["research_id"]: record for record in (result.get("retrieval") or {}).get("results", [])}
        checks["historical_status"] = all(retrieved.get(identity, {}).get("metadata", {}).get("status") == status for identity, status in records.items())
    if route == "knowledge_only":
        for identity, status in records.items():
            eid = f"knowledge-{identity}"
            item = evidence.get(eid, {})
            provenance = item.get("provenance", {})
            claim_text = "\n".join(claim["claim"] for claim in claims if eid in claim["evidence_ids"])
            patterns = expected["record_patterns"][identity]
            checks[identity] = (
                item.get("evidence_type") == "knowledge_record" and item.get("research_id") == identity
                and bool(item.get("content")) and provenance.get("record_status") == status
                and bool(provenance.get("source_path")) and bool(provenance.get("record_date"))
                and re.fullmatch(r"[0-9a-f]{64}", provenance.get("source_hash") or "") is not None
                and f"[{eid}]" in answer and _patterns(answer, patterns) and _patterns(claim_text, patterns)
            )
        checks["knowledge_evidence_only"] = all(item.get("evidence_type") == "knowledge_record" for item in evidence.values())
        return checks
    if route in {"needs_input", "safety_blocked"}:
        checks["no_evidence"] = not evidence and result.get("grounding") is None
        if route == "needs_input":
            checks["minimal_clarification"] = len(answer) <= 600 and not re.search(r"(?:请|提供|supply|provide)[\s\S]{0,40}(?:API|CSV|内部路径|repo path)", answer, re.I)
        else:
            checks["answer"] = not answer  # Current blocked runtime emits no secret-bearing text.
        return checks

    steps = observed["steps"]
    if len(steps) != 1 or steps[0]["name"] != route:
        return {**checks, "executed_tool": False}
    step = steps[0]
    checks["executed_tool"] = step["status"] == "success"
    eid = f"step-1-{route}"
    checks["tool_citation"] = eid in evidence and f"[{eid}]" in answer and any(eid in claim["evidence_ids"] for claim in claims)
    checks["fresh_evidence_only"] = bool(evidence) and all(key.endswith(f"-{route}") and key.startswith("step-") for key in evidence)
    output = evidence.get(eid, {})
    args = step["arguments"]
    wanted = expand_tokens(expected.get("arguments", {}), {f"${{{key}}}": value for key, value in artifacts.items()})
    for key, value in wanted.items():
        actual = args.get(key)
        if key in ARTIFACTS:
            checks[f"argument.{key}"] = isinstance(actual, str) and Path(actual).resolve() == Path(value).resolve()
        elif key == "factor":
            checks["argument.factor"] = isinstance(actual, str) and Path(actual).stem.casefold() == value.casefold()
        elif key == "ic_horizons":
            checks[f"argument.{key}"] = isinstance(actual, list) and sorted(actual) == sorted(value)
        else:
            checks[f"argument.{key}"] = actual == value and not (isinstance(actual, bool) and not isinstance(value, bool))
    if route == "inspect_universe":
        counts = next((row for row in output.get("daily_counts", []) if row.get("date") == wanted["start_date"]), {})
        checks["dated_counts"] = all(type(counts.get(key)) is int and counts[key] >= 0 for key in ("eligible", "trading", "buyable", "sellable"))
        checks["answer_counts"] = checks["dated_counts"] and all(re.search(rf"(?<!\d){counts[key]}(?!\d)", answer.replace(",", "")) for key in ("trading", "buyable", "sellable"))
    elif route == "evaluate_factor":
        checks["fresh_factor"] = output.get("factor", {}).get("name", "").casefold() == "high52"
        for horizon in ("5", "20"):
            group = output.get("groups", {}).get(horizon, {})
            checks[f"H{horizon}"] = output.get("ic", {}).get(horizon, {}).get("n_days", 0) > 0 and group.get("n_groups") == 5 and len(group.get("groups", [])) == 5 and any(row.get("n", 0) > 0 and _has_number(row.get("mean")) for row in group["groups"])
    elif route == "run_backtest":
        config = output.get("config", {})
        checks["canonical_T1_costs"] = config.get("t_plus_1") is True and config.get("buy_cost") == .001 and config.get("sell_cost") == .0015 and step.get("provenance", {}).get("t_plus_1_owned_by") == "research.vector_backtest"
        checks["backtest_metrics"] = output.get("observation", {}).get("trading_days", 0) > 0 and _has_number(output.get("metrics", {}).get("net"))
    else:
        spec, spec_error = validate_experiment_spec(args.get("spec"))
        text = json.dumps(spec, ensure_ascii=False)
        checks["structured_spec"] = spec_error is None and all(term in text for term in expected.get("spec_terms", [])) and _patterns(text, expected.get("spec_patterns", []))
        value, result_error = validate_experiment_result(output.get("result"))
        provenance = output.get("provenance", {})
        _, provenance_error = validate_experiment_provenance({key: provenance.get(key) for key in EXPERIMENT_PROVENANCE_FIELDS})
        checks["experiment_contract"] = result_error is None and provenance_error is None and _has_number(value["metrics"]) and any(count > 0 for count in value["sample_counts"].values())
        checks["isolated_validated"] = provenance.get("executor") == "unshare-chroot-v1" and set(provenance.get("validation_status", {}).values()) == {"passed"} and bool(provenance.get("source_sha256")) and provenance.get("repair_attempts", 0) in (0, 1)
        checks["actual_bounds"] = bool(value) and provenance.get("actual_data_bounds") == {"start": value["data_coverage"].get("actual_start"), "end": value["data_coverage"].get("actual_end")} and all(provenance["actual_data_bounds"].values())
    return checks


def score_result(case: dict[str, Any], result: dict[str, Any], *, artifacts: dict[str, str] | None = None, usefulness_0_2: int | None = None) -> dict[str, Any]:
    """Deterministic acceptance gates + existing grounding, not a semantic judge."""
    _usefulness(usefulness_0_2)
    observed, route = result.get("observed") or {}, case["expected_route"]
    summary = (result.get("telemetry") or {}).get("summary") or {}
    outcome = (observed.get("outcome") or {}).get("status", "error")
    planned = (observed.get("planning") or {}).get("steps", [])
    steps, run = observed.get("steps", []), _run_dict(result)
    names = [step["name"] for step in planned]
    executed = [step["name"] for step in steps]
    allowed = [route] if route in TOOLS else []
    unnecessary = max(len(actions) - min(actions.count(route), len(allowed)) for actions in (names, executed))
    stages = ((summary.get("per_stage") or {}))
    no_execution = not steps and run is None and not any(stage in stages for stage in ("hitl", "experiment_authoring"))
    if route in TOOLS:
        route_correct = names == allowed and not unnecessary
    elif route == "safety_blocked":
        route_correct = (result.get("safety") or {}).get("status") == "blocked" and observed.get("planning") is None and no_execution and summary.get("calls") == 0
    else:
        route_correct = (observed.get("planning") or {}).get("status") == ("no_action" if route == "knowledge_only" else "needs_input") and no_execution and not names
    failure_stage = summary.get("failure_stage") or result.get("error_stage")
    grounding = result.get("grounding")
    grounded = None if grounding is None else False
    checks = {"route": bool(route_correct), "outcome": outcome == case["expected"]["outcome"]}
    try:
        if grounding:
            _, error = parse_grounding_response(json.dumps({"answer": grounding["answer"], "claims": grounding["claims"]}), {item["id"] for item in result.get("evidence") or []})
            grounded = not error and grounding["answer"] == result.get("answer") and bool(grounding["claims"]) and grounding.get("fully_grounded") is True and all(claim["grounding"] == "supported" for claim in grounding["claims"])
        evidence = result.get("evidence") or []
        checks["evidence_contract"] = all(set(item) == {"id", "text"} and isinstance(item["id"], str) and isinstance(item["text"], str) for item in evidence) and len(evidence) == len({item["id"] for item in evidence})
        checks["grounding"] = grounded is True if route in TOOLS | {"knowledge_only"} else grounding is None
        checks["lifecycle"] = (run is not None and run.get("status") == "completed" and run.get("final_status") == "success") if route in TOOLS else run is None
        checks["safety"] = failure_stage == "safety" if route == "safety_blocked" else (result.get("safety") or {}).get("status") == "allowed"
        checks.update(_task_checks(case, result, artifacts or {}))
    except (KeyError, TypeError, ValueError, AttributeError):
        checks["result_contract"] = False
    task_success = all(value for key, value in checks.items() if key != "route")
    return {
        "case": case["id"], "expected_route": route, "terminal_outcome": outcome,
        "task_success": bool(task_success), "route_correct": bool(route_correct), "grounded": grounded,
        "unnecessary_action_count": unnecessary, "human_intervention": outcome in {"needs_input", "needs_approval"} or bool((run or {}).get("human_intervention")),
        "usefulness_0_2": usefulness_0_2, "provider_calls": summary.get("calls"), "tokens": summary.get("total_tokens"),
        "estimated_cost": summary.get("estimated_cost"), "estimated_cost_currency": summary.get("estimated_cost_currency"),
        "wall_clock_ms": summary.get("wall_clock_ms"), "failure_stage": failure_stage,
        "case_pass": all(checks.values()), "failed_checks": [key for key, value in checks.items() if not value],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["case_pass"] is not None]
    def total(key):
        values = [row[key] for row in scored]
        return sum(values) if values and all(value is not None for value in values) else None
    currencies = {row["estimated_cost_currency"] for row in scored}
    ratings = [row["usefulness_0_2"] for row in rows if row["usefulness_0_2"] is not None]
    rates = {}
    for key in ("task_success", "route_correct", "grounded"):
        values = [row[key] for row in scored if row[key] is not None]
        rates[f"{key}_rate"] = sum(values) / len(values) if values else None
    return {
        **rates,
        "selected": len(rows), "executed": len(scored), "not_run": len(rows) - len(scored),
        "passed": sum(row["case_pass"] for row in scored), "failed": sum(not row["case_pass"] for row in scored),
        "pass_rate": sum(row["case_pass"] for row in scored) / len(scored) if scored else None,
        "provider_calls": total("provider_calls"), "tokens": total("tokens"),
        "estimated_cost": total("estimated_cost") if len(currencies) == 1 else None,
        "estimated_cost_currency": next(iter(currencies)) if len(currencies) == 1 else None,
        "wall_clock_ms": total("wall_clock_ms"), "unnecessary_action_count": total("unnecessary_action_count"),
        "human_intervention": sum(row["human_intervention"] for row in rows),
        "rated": len(ratings), "mean_usefulness_0_2": sum(ratings) / len(ratings) if ratings else None,
        "failure_stages": dict(Counter(row["failure_stage"] for row in rows if row["failure_stage"])),
    }


def run_suite(*, cases=CASES, runtime=None, runtime_options=None, artifacts=None, usefulness=None) -> dict[str, Any]:
    """Inject runtime/clients for tests; expected labels never enter model inputs."""
    validate_cases(cases)
    artifacts, usefulness = dict(artifacts or {}), dict(usefulness or {})
    if set(artifacts) - ARTIFACTS or any(not isinstance(value, str) or not value.strip() for value in artifacts.values()):
        raise ValueError("artifacts must be nonempty runtime paths for weights/close/open")
    if set(usefulness) - {case["id"] for case in cases}:
        raise ValueError("manual ratings must refer to selected cases")
    for rating in usefulness.values():
        _usefulness(rating)
    rows = []
    for case in cases:
        rating = usefulness.get(case["id"], case.get("usefulness_0_2"))
        options = dict(runtime_options or {})
        missing = set(case.get("required_artifacts", [])) - set(artifacts)
        if missing:
            row = {key: None for key in ("task_success", "route_correct", "grounded", "provider_calls", "tokens", "estimated_cost", "estimated_cost_currency", "wall_clock_ms", "unnecessary_action_count", "case_pass")}
            row.update(case=case["id"], expected_route=case["expected_route"], terminal_outcome="not_run", human_intervention=True, usefulness_0_2=rating, failure_stage="precondition", failed_checks=[f"missing_artifact.{key}" for key in sorted(missing)])
            rows.append(row)
            continue
        if case.get("required_artifacts"):
            options["context_items"] = [*options.get("context_items", []), {"id": "uat_artifacts", "kind": "runtime_truth", "text": json.dumps(artifacts, ensure_ascii=False)}]
        tick = perf_counter()
        try:
            result = (runtime or run_request)(case["request"], **options)
            row = score_result(case, result, artifacts=artifacts, usefulness_0_2=rating)
            row["runtime"] = {key: result.get(key) for key in ("answer", "observed", "telemetry", "evidence", "grounding", "synthesis", "error_type", "error_stage")}
        except Exception as exc:
            row = {key: None for key in ("grounded", "provider_calls", "tokens", "estimated_cost", "estimated_cost_currency", "unnecessary_action_count")}
            row.update(case=case["id"], expected_route=case["expected_route"], terminal_outcome="error", task_success=False, route_correct=False, human_intervention=False, usefulness_0_2=rating, wall_clock_ms=(perf_counter() - tick) * 1000, failure_stage="harness", case_pass=False, failed_checks=["runtime_exception"], error_type=type(exc).__name__)
        rows.append(row)
    return {"schema_version": "product-uat-v1", "cases": rows, "summary": aggregate(rows)}


def print_report(report: dict[str, Any]) -> None:
    def cell(value):
        return "-" if value is None else "Y" if value is True else "N" if value is False else f"{value:.6g}" if isinstance(value, float) else str(value)
    columns = (
        ("case", "Case"), ("expected_route", "Expected route"), ("terminal_outcome", "Outcome"),
        ("task_success", "Task"), ("route_correct", "Route"), ("grounded", "Grnd"),
        ("unnecessary_action_count", "Extra"), ("human_intervention", "Human"), ("usefulness_0_2", "Use"),
        ("provider_calls", "Calls"), ("tokens", "Tokens"), ("estimated_cost", "Cost"),
        ("wall_clock_ms", "ms"), ("failure_stage", "Failure"), ("case_pass", "Pass"),
    )
    table = [[title for _, title in columns]]
    for row in report["cases"]:
        values = [cell(row[key]) for key, _ in columns]
        if row["estimated_cost"] is not None:
            values[11] += row["estimated_cost_currency"] or "?"
        table.append(values)
    widths = [max(len(row[index]) for row in table) for index in range(len(columns))]
    for row in table:
        print(" ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
    summary = report["summary"]
    print(f"Passed {summary['passed']}/{summary['executed']} executed; selected={summary['selected']}, not_run={summary['not_run']}; calls={cell(summary['provider_calls'])}, tokens={cell(summary['tokens'])}, cost={cell(summary['estimated_cost'])} {summary['estimated_cost_currency'] or ''}, wall_ms={cell(summary['wall_clock_ms'])}; manual usefulness={cell(summary['mean_usefulness_0_2'])} ({summary['rated']} rated)")
    print(f"Core task rate={cell(summary['task_success_rate'])}, route rate={cell(summary['route_correct_rate'])}, grounded rate={cell(summary['grounded_rate'])}; extra actions={cell(summary['unnecessary_action_count'])}, human intervention={summary['human_intervention']}")
    print("Failures/not-run: " + (", ".join(row["case"] for row in report["cases"] if row["case_pass"] is not True) or "none") + "; full checks/trace/evidence: --json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("deepseek", "openai"), required=True, help="explicitly opt into real provider calls")
    parser.add_argument("--model", default="")
    parser.add_argument("--retrieval", choices=("qdrant", "none", "ollama", "openai"), default="qdrant")
    parser.add_argument("--retrieval-strategy", choices=("dense", "hybrid"), default="dense")
    parser.add_argument("--case", action="append", choices=[case["id"] for case in CASES])
    for flag in ("weights", "close", "open"):
        parser.add_argument(f"--{flag}")
    parser.add_argument("--usefulness", action="append", default=[], metavar="UAT-NN=0|1|2")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        usefulness = {}
        for item in args.usefulness:
            identity, value = item.split("=", 1)
            if identity in usefulness:
                raise ValueError("duplicate manual rating")
            usefulness[identity] = int(value)
        artifacts = {key: value for key, value in zip(("target_weights", "price_panel", "open_panel"), (args.weights, args.close, args.open)) if value is not None}
        report = run_suite(cases=[case for case in CASES if not args.case or case["id"] in args.case], artifacts=artifacts, usefulness=usefulness,
                           runtime_options={"provider": args.provider, "model": args.model, "retrieval": args.retrieval, "retrieval_strategy": args.retrieval_strategy})
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0 if all(row["case_pass"] is True for row in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
