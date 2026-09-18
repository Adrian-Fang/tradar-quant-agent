"""Minimal integrated runtime for one research request."""

from __future__ import annotations

import json
from typing import Any

from .answer.synthesizer import synthesize_answer
from .context.builder import construct_context
from .context.selector import select_context
from .core.contracts import ResearchRun
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


def tool_results_to_evidence(tool_results: list[Any]) -> list[dict[str, str]]:
    """Serialize successful tool outputs as grounding-compatible evidence."""
    evidence = []
    for seq, tool_result in enumerate(tool_results, 1):
        if tool_result.status not in {"success", "partial"}:
            continue
        text = json.dumps(
            tool_result.to_dict()["result"],
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
) -> dict[str, Any]:
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

    items = [
        {"id": "request_scope", "kind": "current_instruction", "text": user_request},
        *(context_items or []),
    ]
    selected = select_context(items)["selected"]
    retrieval_result = None
    retrieval_trace = {"status": "not_used", "research_ids": []}

    if semantic_embedder is not None:
        if retrieval_client is None:
            raise ValueError("retrieval_client is required with semantic_embedder")
        retrieval_result = retrieve_verified(
            user_request,
            client=retrieval_client,
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
            items.append(item)
        selected = select_context(items)["selected"]

    construction = construct_context(
        user_request,
        {"status": "ok", "context": selected},
    )
    planned = plan_request(construction["input"], client=planner_client, model=model)
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
        )

    if hitl_client is None:
        raise ValueError("hitl_client is required for a ready plan")
    gate = gate_action(
        user_request,
        proposed_action or {
            "description": "Execute the validated research plan.",
            "environment": "local",
            "reversible": True,
        },
        existing_approval or {"approved": False, "scope": None},
        product_boundaries or [],
        client=hitl_client,
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
            orchestration={"status": "error", "type": "orchestration_error"},
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
        )

    bound_evidence = tool_results_to_evidence(tool_results)
    synthesis = None
    synthesized_answer = None
    if synthesis_client is not None:
        synthesis_result = synthesize_answer(
            user_request,
            bound_evidence,
            client=synthesis_client,
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
            answer, evidence, client=grounding_client, model=model,
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
    )


__all__ = ["run_agent", "tool_results_to_evidence"]
