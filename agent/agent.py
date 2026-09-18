"""Minimal integrated runtime for one research request."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .answer.synthesizer import synthesize_answer
from .context.builder import construct_context
from .context.selector import select_context
from .core.contracts import ResearchRun
from .core.safety import SafetyClient, check_request_safety, quarantine_untrusted_text
from .core.telemetry import RunTelemetry, TelemetryClient
from .grounding.verifier import verify_answer_grounding
from .hitl.gate import gate_action
from .planning.planner import plan_request
from .retrieval.relevance_verifier import retrieve_verified
from .tools.executor import execute_steps


def _steps(run: ResearchRun) -> list[dict[str, Any]]:
    return [
        {
            "name": step["tool_name"],
            "arguments": step["normalized_args"],
            "status": step["status"],
        }
        for step in run.steps
    ]


def _bounded_array(values: list[Any], limit: int) -> Any:
    if len(values) <= limit:
        return values
    head = limit // 2
    return {
        "items": values[:head] + values[-head:],
        "omitted_count": len(values) - (head * 2),
    }


def _bounded_tool_output(tool_name: str, result: Any) -> Any:
    if tool_name != "inspect_universe" or not isinstance(result, Mapping):
        return result

    output = dict(result)
    if isinstance(output.get("daily_counts"), list):
        output["daily_counts"] = _bounded_array(output["daily_counts"], 24)

    snapshots = output.get("snapshots")
    if isinstance(snapshots, list):
        if len(snapshots) > 8:
            head = 4
            snapshot_items = snapshots[:head] + snapshots[-head:]
            snapshot_output = {
                "items": snapshot_items,
                "omitted_count": len(snapshots) - (head * 2),
            }
        else:
            snapshot_items = snapshots
            snapshot_output = None

        bounded_snapshots = []
        for snapshot in snapshot_items:
            if not isinstance(snapshot, Mapping):
                bounded_snapshots.append(snapshot)
                continue
            snapshot_copy = dict(snapshot)
            membership = snapshot_copy.get("membership")
            if isinstance(membership, Mapping):
                membership_copy = dict(membership)
                for name in ("eligible", "trading", "buyable", "sellable", "price_limit_known"):
                    if isinstance(membership_copy.get(name), list):
                        membership_copy[name] = _bounded_array(membership_copy[name], 24)
                snapshot_copy["membership"] = membership_copy
            bounded_snapshots.append(snapshot_copy)

        if snapshot_output is None:
            output["snapshots"] = bounded_snapshots
        else:
            snapshot_output["items"] = bounded_snapshots
            output["snapshots"] = snapshot_output
    return output


def tool_results_to_evidence(tool_results: list[Any]) -> list[dict[str, str]]:
    """Serialize successful tool outputs as grounding-compatible evidence."""
    evidence = []
    for seq, tool_result in enumerate(tool_results, 1):
        if tool_result.status not in {"success", "partial"}:
            continue
        text = json.dumps(
            _bounded_tool_output(tool_result.tool_name, tool_result.to_dict()["result"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        evidence.append({
            "id": f"step-{seq}-{tool_result.tool_name}",
            "text": text,
        })
    return evidence


def _finish(
    *,
    context_ids: list[str],
    planning: dict[str, Any] | None,
    steps: list[dict[str, Any]],
    retrieval: dict[str, Any],
    research_run: ResearchRun | None,
    grounding: dict[str, Any] | None,
    grounding_trace: dict[str, Any] | None = None,
    hitl: dict[str, Any] | None,
    outcome: str,
    plan: dict[str, Any] | None = None,
    retrieval_result: dict[str, Any] | None = None,
    status: str = "ok",
    error_type: str | None = None,
    error: str = "",
    error_stage: str | None = None,
    orchestration: dict[str, Any] | None = None,
    answer: str | None = None,
    evidence: list[dict[str, str]] | None = None,
    synthesis: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
    telemetry: RunTelemetry | None = None,
) -> dict[str, Any]:
    if telemetry is not None:
        runtime_stage = error_stage
        if runtime_stage is None and outcome in {"needs_input", "no_action"}:
            runtime_stage = "planning"
        elif runtime_stage is None and outcome == "needs_approval":
            runtime_stage = "hitl"
        elif runtime_stage is None and outcome == "abstain":
            runtime_stage = (
                "synthesis"
                if synthesis and synthesis.get("status") == "insufficient_evidence"
                else "retrieval"
                if retrieval.get("status") == "abstain"
                else None
            )
        elif runtime_stage is None and outcome == "blocked":
            runtime_stage = (
                "safety"
                if safety and safety.get("status") == "blocked"
                else "hitl"
                if hitl and hitl.get("decision") != "proceed"
                else "grounding"
                if grounding_trace and not grounding_trace.get("fully_grounded")
                else None
            )
        elif runtime_stage is None and outcome == "error":
            runtime_stage = (
                "execution"
                if research_run and research_run.status == "failed"
                else "planning"
                if planning and planning.get("status") == "error"
                else "retrieval"
                if retrieval.get("status") == "error"
                else None
            )
        telemetry.set_runtime_failure_stage(runtime_stage)
    observed = {
        "context": {"selected_ids": context_ids},
        "planning": planning,
        "steps": steps,
        "retrieval": retrieval,
        "research_run": (
            {"status": research_run.status, "final_status": research_run.final_status}
            if research_run is not None else None
        ),
        "grounding": grounding_trace if grounding_trace is not None else grounding,
        "hitl": hitl,
        "orchestration": orchestration,
        "outcome": {"status": outcome},
    }
    return {
        "status": status,
        "plan": plan,
        "retrieval": retrieval_result,
        "research_run": research_run,
        "grounding": grounding,
        "hitl": hitl,
        "observed": observed,
        "answer": answer,
        "evidence": evidence,
        "synthesis": synthesis,
        "safety": safety,
        "telemetry": telemetry.envelope() if telemetry is not None else None,
        "error_type": error_type,
        "error_stage": error_stage,
        "error": error,
    }


def run_agent(
    user_request: str,
    *,
    planner_client: Any,
    hitl_client: Any | None = None,
    synthesis_client: Any | None = None,
    grounding_client: Any | None = None,
    retrieval_client: Any | None = None,
    semantic_embedder: Any | None = None,
    prepared_corpus: dict[str, Any] | None = None,
    context_items: list[dict[str, Any]] | None = None,
    proposed_action: dict[str, Any] | None = None,
    existing_approval: dict[str, Any] | None = None,
    product_boundaries: list[str] | None = None,
    answer: str | None = None,
    evidence: list[dict[str, str]] | None = None,
    model: str = "",
    run_id: str | None = None,
    candidate_limit: int = 5,
    market: str | None = None,
    topic: str | None = None,
    record_status: str | None = None,
) -> dict[str, Any]:
    """Run context/retrieval, planning, HITL, execution, synthesis, and grounding."""
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("user_request must be a non-empty string")

    telemetry = RunTelemetry()
    action = proposed_action or {
        "description": "Execute the validated research plan.",
        "environment": "local",
        "reversible": True,
    }
    boundaries = product_boundaries or []
    safety = check_request_safety(user_request, action, boundaries)
    if safety["status"] == "blocked":
        return _finish(
            context_ids=["request_scope"],
            planning=None,
            steps=[],
            retrieval={"status": "not_used", "research_ids": []},
            research_run=None,
            grounding=None,
            hitl=None,
            outcome="blocked",
            safety=safety,
            telemetry=telemetry,
        )

    safe_context_items = []
    for item in context_items or []:
        event = quarantine_untrusted_text(
            item.get("text", ""), source=f"context:{item['id']}"
        )
        if event["status"] == "quarantined":
            safety["events"].append(event)
        else:
            safe_context_items.append(item)

    items = [
        {"id": "request_scope", "kind": "current_instruction", "text": user_request},
        *safe_context_items,
    ]
    selected = select_context(items)["selected"]
    retrieval_result = None
    retrieval_trace = {"status": "not_used", "research_ids": []}

    if semantic_embedder is not None:
        if retrieval_client is None:
            raise ValueError("retrieval_client is required with semantic_embedder")
        retrieval_result = retrieve_verified(
            user_request,
            client=TelemetryClient(
                SafetyClient(retrieval_client),
                telemetry,
                stage="retrieval_verifier",
                model=model,
            ),
            model=model,
            candidate_limit=candidate_limit,
            market=market,
            topic=topic,
            status=record_status,
            embedder=semantic_embedder,
            prepared_corpus=prepared_corpus,
        )
        retrieval_trace = {
            "status": retrieval_result["status"],
            "research_ids": [
                item["research_id"] for item in retrieval_result.get("results", [])
            ],
        }
        if retrieval_result["status"] == "error":
            error = retrieval_result.get("errors", [{}])[0]
            return _finish(
                context_ids=[item["id"] for item in selected],
                planning=None,
                steps=[],
                retrieval=retrieval_trace,
                research_run=None,
                grounding=None,
                hitl=None,
                outcome="error",
                retrieval_result=retrieval_result,
                status="error",
                error_type=error.get("error_type", "retrieval_error"),
                error=error.get("error", "retrieval failed"),
                error_stage="retrieval",
                safety=safety,
                telemetry=telemetry,
            )
        if retrieval_result["status"] == "abstain":
            return _finish(
                context_ids=[item["id"] for item in selected],
                planning=None,
                steps=[],
                retrieval=retrieval_trace,
                research_run=None,
                grounding=None,
                hitl=None,
                outcome="abstain",
                retrieval_result=retrieval_result,
                safety=safety,
                telemetry=telemetry,
            )
        for result in retrieval_result["results"]:
            item = {
                "id": result["research_id"],
                "kind": "retrieved_knowledge",
                "text": result["text"],
            }
            for field in ("source", "provenance"):
                if result.get(field):
                    item[field] = result[field]
            event = quarantine_untrusted_text(
                item["text"], source=f"retrieval:{item['id']}"
            )
            if event["status"] == "quarantined":
                safety["events"].append(event)
            else:
                items.append(item)
        selected = select_context(items)["selected"]

    construction = construct_context(
        user_request,
        {"status": "ok", "context": selected},
    )
    planned = plan_request(
        construction["input"],
        client=TelemetryClient(
            SafetyClient(planner_client), telemetry, stage="planning", model=model
        ),
        model=model,
    )
    if planned["status"] == "error":
        planning = {
            "status": "error",
            "steps": [],
            "error_type": planned["error_type"],
        }
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=[],
            retrieval=retrieval_trace,
            research_run=None,
            grounding=None,
            hitl=None,
            outcome="error",
            status="error",
            error_type=planned["error_type"],
            error=planned["error"],
            error_stage="planning",
            safety=safety,
            telemetry=telemetry,
        )

    plan = planned["plan"]
    planning = {"status": plan["status"], "steps": plan["steps"]}
    if plan["status"] != "ready":
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=[],
            retrieval=retrieval_trace,
            research_run=None,
            grounding=None,
            hitl=None,
            outcome=plan["status"],
            plan=plan,
            retrieval_result=retrieval_result,
            safety=safety,
            telemetry=telemetry,
        )

    if hitl_client is None:
        raise ValueError("hitl_client is required for a ready plan")
    gate = gate_action(
        user_request,
        action,
        existing_approval or {"approved": False, "scope": None},
        boundaries,
        client=TelemetryClient(
            SafetyClient(hitl_client), telemetry, stage="hitl", model=model
        ),
        model=model,
    )
    if gate["status"] == "error":
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=[],
            retrieval=retrieval_trace,
            research_run=None,
            grounding=None,
            hitl=None,
            outcome="error",
            plan=plan,
            retrieval_result=retrieval_result,
            status="error",
            error_type=gate["error_type"],
            error=gate["error"],
            error_stage="hitl",
            safety=safety,
            telemetry=telemetry,
        )

    hitl_trace = {"decision": gate["decision"]}
    if gate["decision"] != "proceed":
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=[],
            retrieval=retrieval_trace,
            research_run=None,
            grounding=None,
            hitl=hitl_trace,
            outcome=gate["decision"],
            plan=plan,
            retrieval_result=retrieval_result,
            safety=safety,
            telemetry=telemetry,
        )

    run = ResearchRun(
        user_request=user_request,
        **({"run_id": run_id} if run_id is not None else {}),
    )
    tool_results = []
    try:
        run = execute_steps(plan["steps"], run=run, tool_results=tool_results)
    except Exception as exc:
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=_steps(run),
            retrieval=retrieval_trace,
            research_run=run,
            grounding=None,
            hitl=hitl_trace,
            outcome="error",
            plan=plan,
            retrieval_result=retrieval_result,
            status="error",
            error_type="orchestration_error",
            error=f"{type(exc).__name__}: {exc}",
            error_stage="orchestration",
            orchestration={"status": "error", "type": "orchestration_error"},
            safety=safety,
            telemetry=telemetry,
        )

    steps = _steps(run)
    if run.status == "failed":
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=steps,
            retrieval=retrieval_trace,
            research_run=run,
            grounding=None,
            hitl=hitl_trace,
            outcome="error",
            plan=plan,
            retrieval_result=retrieval_result,
            error_stage="execution",
            safety=safety,
            telemetry=telemetry,
        )

    bound_evidence = tool_results_to_evidence(tool_results)
    safe_evidence = []
    tool_injection = False
    for item in bound_evidence:
        event = quarantine_untrusted_text(
            item["text"], source=f"tool_output:{item['id']}"
        )
        if event["status"] == "quarantined":
            tool_injection = True
            safety["events"].append(event)
        else:
            safe_evidence.append(item)
    bound_evidence = safe_evidence
    if tool_injection and not bound_evidence:
        safety.update({
            "status": "blocked",
            "rule": "untrusted_instruction_injection",
            "reason": "all executable evidence was quarantined by the safety boundary",
        })
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=steps,
            retrieval=retrieval_trace,
            research_run=run,
            grounding=None,
            hitl=hitl_trace,
            outcome="blocked",
            plan=plan,
            retrieval_result=retrieval_result,
            safety=safety,
            telemetry=telemetry,
        )
    synthesis = None
    synthesized_answer = None
    if synthesis_client is not None:
        synthesis_result = synthesize_answer(
            user_request,
            bound_evidence,
            client=TelemetryClient(
                SafetyClient(synthesis_client),
                telemetry,
                stage="synthesis",
                model=model,
            ),
            model=model,
        )
        if synthesis_result["status"] == "error":
            return _finish(
                context_ids=[item["id"] for item in selected],
                planning=planning,
                steps=steps,
                retrieval=retrieval_trace,
                research_run=run,
                grounding=None,
                hitl=hitl_trace,
                outcome="error",
                plan=plan,
                retrieval_result=retrieval_result,
                status="error",
                error_type=synthesis_result["error_type"],
                error_stage="synthesis",
                error=synthesis_result["error"],
                evidence=bound_evidence,
                safety=safety,
                telemetry=telemetry,
            )
        synthesis = synthesis_result["result"]
        synthesized_answer = synthesis["answer"]
        cited_ids = set(synthesis["evidence_ids"])
        bound_evidence = [item for item in bound_evidence if item["id"] in cited_ids]
        if synthesis["status"] == "insufficient_evidence":
            return _finish(
                context_ids=[item["id"] for item in selected],
                planning=planning,
                steps=steps,
                retrieval=retrieval_trace,
                research_run=run,
                grounding=None,
                hitl=hitl_trace,
                outcome="abstain",
                plan=plan,
                retrieval_result=retrieval_result,
                answer=synthesized_answer,
                evidence=bound_evidence,
                synthesis=synthesis,
                safety=safety,
                telemetry=telemetry,
            )
    elif answer is None and evidence is None:
        raise ValueError("synthesis_client is required after successful execution")

    grounding = None
    grounding_trace = None
    if synthesis_client is not None:
        answer = synthesized_answer
        evidence = bound_evidence
    if answer is not None:
        if grounding_client is None or evidence is None:
            raise ValueError("grounding_client and evidence are required with answer")
        grounding_result = verify_answer_grounding(
            answer,
            evidence,
            client=TelemetryClient(
                SafetyClient(grounding_client),
                telemetry,
                stage="grounding",
                model=model,
            ),
            model=model,
        )
        if grounding_result["status"] == "error":
            return _finish(
                context_ids=[item["id"] for item in selected],
                planning=planning,
                steps=steps,
                retrieval=retrieval_trace,
                research_run=run,
                grounding=None,
                hitl=hitl_trace,
                outcome="error",
                plan=plan,
                retrieval_result=retrieval_result,
                status="error",
                error_type=grounding_result["error_type"],
                error_stage="grounding",
                error=grounding_result["error"],
                answer=answer,
                evidence=evidence,
                synthesis=synthesis,
                safety=safety,
                telemetry=telemetry,
            )
        grounding = grounding_result["assessment"]
        grounding_trace = {
            "fully_grounded": grounding["fully_grounded"],
            "labels": [claim["grounding"] for claim in grounding["claims"]],
        }

    outcome = (
        "blocked"
        if grounding_trace and not grounding_trace["fully_grounded"]
        else "success"
    )
    return _finish(
        context_ids=[item["id"] for item in selected],
        planning=planning,
        steps=steps,
        retrieval=retrieval_trace,
        research_run=run,
        grounding=grounding,
        grounding_trace=grounding_trace,
        hitl=hitl_trace,
        outcome=outcome,
        plan=plan,
        retrieval_result=retrieval_result,
        answer=answer,
        evidence=evidence,
        synthesis=synthesis,
        safety=safety,
        telemetry=telemetry,
    )


__all__ = ["run_agent", "tool_results_to_evidence"]
