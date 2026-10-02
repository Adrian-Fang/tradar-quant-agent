"""Minimal integrated runtime for one research request."""

from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from .answer.synthesizer import synthesize_answer
from .context.builder import construct_context
from .context.selector import select_context
from .core.contracts import ResearchRun, ToolResult
from .core.safety import SafetyClient, check_request_safety, quarantine_untrusted_text
from .core.telemetry import RunTelemetry, TelemetryClient
from .core.resources import load_json
from .grounding.verifier import verify_answer_grounding
from .hitl.gate import gate_action
from .loop.runner import run_loop
from .planning.planner import plan_request
from .retrieval.hybrid_retriever import KnowledgeRetriever
from .retrieval.qdrant_store import filter_spec
from .retrieval.relevance_verifier import verify_candidates
from .retrieval.semantic_retriever import retrieve_semantic
from .tools.experiment import author_experiment, run_research_experiment


CAPABILITY_ITEMS = load_json("capabilities.json")
_CAPABILITY_MARKERS = (
    "你能做什么",
    "你有哪些数据",
    "支持哪些研究",
    "可以评估哪些 factor",
    "可以评估哪些因子",
    "数据覆盖",
    "数据质量",
    "有什么限制",
    "what can you do",
    "what data",
    "data quality",
    "what are your limitations",
)


def _is_capability_request(user_request: str) -> bool:
    text = user_request.casefold()
    return any(marker.casefold() in text for marker in _CAPABILITY_MARKERS)


def _clarification_answer(reason: Any) -> str:
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return "请补充完成这项研究所需的最少信息。"


def _capability_answer() -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    evidence = [{"id": item["id"], "text": item["text"]} for item in CAPABILITY_ITEMS]
    answer = "当前能力与边界：\n" + "\n".join(
        f"- {item['text']}" for item in evidence
    )
    grounding = {
        "answer": answer,
        "claims": [
            {
                "claim": item["text"],
                "evidence_ids": [item["id"]],
                "grounding": "supported",
            }
            for item in evidence
        ],
        "fully_grounded": True,
    }
    return answer, evidence, grounding


def _steps(run: ResearchRun) -> list[dict[str, Any]]:
    return [
        {
            "name": step["tool_name"],
            "arguments": step["normalized_args"],
            "status": step["status"],
            "provenance": step.get("provenance", {}),
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


def _planner_loop_decider(
    planning_input: str,
    planner_client: Any,
    telemetry: RunTelemetry,
    model: str,
):
    def decide(observation: dict[str, Any]) -> dict[str, Any]:
        planned = plan_request(
            planning_input,
            client=TelemetryClient(
                SafetyClient(planner_client), telemetry, stage="planning", model=model
            ),
            model=model,
            observations=observation,
        )
        if planned["status"] == "error":
            return {
                "status": "error",
                "error_type": planned["error_type"],
                "error": planned["error"],
                "error_stage": "planning",
            }

        plan = planned["plan"]
        if plan["status"] != "ready":
            return {
                "status": plan["status"],
                "step": None,
                "reason": plan["reason"],
            }

        executed = observation.get("steps", [])
        for candidate in plan["steps"]:
            candidate_args = candidate["arguments"]
            already_executed = any(
                step.get("tool_name") == candidate["name"]
                and (
                    not step.get("normalized_args")
                    or all(
                        step["normalized_args"].get(key) == value
                        for key, value in candidate_args.items()
                    )
                )
                for step in executed
            )
            if not already_executed:
                return {
                    "status": "execute",
                    "step": candidate,
                    "reason": plan["reason"],
                }

        return {"status": "finish", "step": None, "reason": "evidence is sufficient"}

    return decide


def tool_results_to_evidence(tool_results: list[Any]) -> list[dict[str, str]]:
    """Serialize successful tool outputs as grounding-compatible evidence."""
    evidence = []
    for seq, tool_result in enumerate(tool_results, 1):
        if tool_result.status not in {"success", "partial"}:
            continue
        source = tool_result.to_dict()
        value = _bounded_tool_output(tool_result.tool_name, source["result"])
        if tool_result.tool_name == "run_research_experiment":
            value = {"result": value, "provenance": source["provenance"]}
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        evidence.append({
            "id": f"step-{seq}-{tool_result.tool_name}",
            "text": text,
        })
    return evidence


def _knowledge_planning_brief(record: dict[str, Any]) -> dict[str, Any]:
    """Bounded related-research context for routing/method reuse, never answer evidence."""
    metadata, sections = record.get("metadata", {}), record.get("sections", {})
    brief = {
        "context_type": "related_research_context", "research_id": record["research_id"],
        "scope": {key: metadata.get(key) for key in ("date", "market", "status")},
        "truncated_fields": [],
    }
    for field, value in brief["scope"].items():
        if isinstance(value, str) and len(value) > 80:
            brief["scope"][field] = value[:80]
            brief["truncated_fields"].append(f"scope.{field}")
    for field, value, limit in (
        ("title", record.get("title", ""), 100),
        ("question", record.get("question", sections.get("Research Question", "")), 240),
        ("method", sections.get("Method", ""), 160),
        ("conclusion", sections.get("Conclusion", ""), 200),
        ("caveats", sections.get("Caveats", ""), 160),
    ):
        brief[field] = value[:limit]
        if len(value) > limit:
            brief["truncated_fields"].append(field)
    return brief


def _knowledge_records_to_evidence(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project verified historical records, never similarity or fresh execution results."""
    evidence, seen = [], set()
    for record in records:
        if record.get("verification", {}).get("supported") is not True:
            continue
        research_id = record["research_id"]
        if research_id in seen:
            continue
        seen.add(research_id)
        metadata = record.get("metadata", {})
        payload = {
            "evidence_type": "knowledge_record", "research_id": research_id,
            "title": record.get("title"), "content": record["text"],
            "provenance": {
                "source_path": record.get("path"), "source_hash": record.get("source_hash"),
                "source_type": record.get("source"), "source_ref": record.get("source_ref", []),
                "record_date": metadata.get("date"), "record_status": metadata.get("status"),
            },
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # ponytail: whole records capped at 12k chars; section-preserving excerpts if the corpus grows.
        if len(text) > 12000:
            raise ValueError(f"knowledge record {research_id} exceeds the 12000-character evidence limit")
        evidence.append({"id": f"knowledge-{research_id}", "text": text})
        if len(evidence) == 5:
            break
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
    loop: dict[str, Any] | None = None,
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
        telemetry.set_runtime_stages(
            runtime_stage if outcome in {"error", "blocked"} else None,
            runtime_stage,
        )
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
    if loop is not None:
        observed["loop"] = loop
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


def _synthesize_and_ground(
    user_request: str, bound_evidence: list[dict[str, str]], *,
    synthesis_client: Any, grounding_client: Any, model: str, telemetry: RunTelemetry,
    answer: str | None = None, evidence: list[dict[str, str]] | None = None,
    conversation_history: list[dict[str, str]] | None = None, **finish_args: Any,
) -> dict[str, Any]:
    """The same answer/evidence contract for knowledge and executed research."""
    finish_args["telemetry"] = telemetry
    synthesis = None
    if synthesis_client is not None:
        result = synthesize_answer(
            user_request, bound_evidence,
            client=TelemetryClient(SafetyClient(synthesis_client), telemetry, stage="synthesis", model=model),
            model=model, conversation_history=conversation_history,
        )
        if result["status"] == "error":
            return _finish(**finish_args, grounding=None, outcome="error", status="error",
                           error_type=result["error_type"], error_stage="synthesis", error=result["error"],
                           evidence=bound_evidence)
        synthesis = result["result"]
        answer = synthesis["answer"]
        cited_ids = set(synthesis["evidence_ids"])
        evidence = [item for item in bound_evidence if item["id"] in cited_ids]
        if synthesis["status"] == "insufficient_evidence":
            return _finish(**finish_args, grounding=None, outcome="abstain", answer=answer,
                           evidence=evidence, synthesis=synthesis)
    elif answer is None and evidence is None:
        raise ValueError("synthesis_client is required to answer from evidence")

    grounding, grounding_trace = None, None
    if answer is not None:
        if grounding_client is None or evidence is None:
            raise ValueError("grounding_client and evidence are required with answer")
        result = verify_answer_grounding(
            answer, evidence,
            client=TelemetryClient(SafetyClient(grounding_client), telemetry, stage="grounding", model=model),
            model=model,
        )
        if result["status"] == "error":
            return _finish(**finish_args, grounding=None, outcome="error", status="error",
                           error_type=result["error_type"], error_stage="grounding", error=result["error"],
                           answer=answer, evidence=evidence, synthesis=synthesis)
        grounding = result["assessment"]
        grounding_trace = {
            "fully_grounded": grounding["fully_grounded"],
            "labels": [claim["grounding"] for claim in grounding["claims"]],
        }
    outcome = "blocked" if grounding_trace and not grounding_trace["fully_grounded"] else "success"
    return _finish(**finish_args, grounding=grounding, grounding_trace=grounding_trace,
                   outcome=outcome, answer=answer, evidence=evidence, synthesis=synthesis)


def run_agent(
    user_request: str,
    *,
    planner_client: Any,
    experiment_authoring_client: Any | None = None,
    hitl_client: Any | None = None,
    synthesis_client: Any | None = None,
    grounding_client: Any | None = None,
    retrieval_client: Any | None = None,
    retrieval_backend: str = "legacy",
    retrieval_strategy: str = "dense",
    knowledge_retriever: Any | None = None,
    retrieval_filters: dict[str, Any] | None = None,
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
    loop_decider: Any | None = None,
    max_iterations: int = 8,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run context/retrieval, iterative planning, HITL, execution, synthesis, and grounding."""
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

    if retrieval_backend not in {"legacy", "qdrant"}:
        raise ValueError("retrieval_backend must be legacy or qdrant")
    if retrieval_strategy not in ("dense", "hybrid"):
        raise ValueError("retrieval_strategy must be dense or hybrid")
    if retrieval_backend == "legacy" and (knowledge_retriever is not None or retrieval_filters):
        raise ValueError("knowledge_retriever and retrieval_filters require the qdrant backend")
    if retrieval_filters is not None and not isinstance(retrieval_filters, dict):
        raise ValueError("retrieval_filters must be an object")
    filters = dict(retrieval_filters or {})
    if retrieval_backend == "qdrant":
        if type(candidate_limit) is not int or not 1 <= candidate_limit <= 5:
            raise ValueError("qdrant candidate_limit must be between 1 and 5")
        for key, value in (("market", market), ("topic", topic), ("status", record_status)):
            if value is not None:
                if key in filters and filters[key] != value:
                    raise ValueError(f"conflicting retrieval filter: {key}")
                filters[key] = value
        filter_spec(filters)

    safe_context_items = []
    for item in context_items or []:
        event = quarantine_untrusted_text(
            item.get("text", ""), source=f"context:{item['id']}"
        )
        if event["status"] == "quarantined":
            safety["events"].append(event)
        else:
            safe_context_items.append(item)

    for index, message in enumerate(conversation_history or [], 1):
        safe_context_items.append({
            "id": f"conversation-{index}",
            "kind": "history",
            "role": message["role"],
            "text": message["content"],
        })

    items = [
        {"id": "request_scope", "kind": "current_instruction", "text": user_request},
        *safe_context_items,
    ]
    selected = select_context(items)["selected"]
    retrieval_result = None
    retrieval_trace = {"status": "not_used", "research_ids": []}
    safe_candidates = []

    if retrieval_backend == "qdrant" or semantic_embedder is not None:
        started = perf_counter()
        details = {"backend": retrieval_backend, "candidate_ids": [], "quarantined_ids": [],
                   "related_research_ids": [], "verified_ids": [], "verification_status": "not_used"}
        if retrieval_backend == "qdrant":
            details.update({"mode": retrieval_strategy, "filters": filters, "candidate_limit": candidate_limit})
        try:
            if retrieval_backend == "qdrant":
                # Construct only after safety checks. Never index or fall back on runtime startup.
                retriever = knowledge_retriever if knowledge_retriever is not None else KnowledgeRetriever()
                search = retriever.search(user_request, mode=retrieval_strategy, limit=candidate_limit, **filters)
                candidates = search["results"]
                details.update({
                    "latency_ms": dict(search.get("latency_ms", {})),
                    "chunks_returned": search.get("chunks_returned"),
                })
            else:
                candidates = retrieve_semantic(
                    user_request, limit=candidate_limit,
                    market=market, topic=topic, status=record_status,
                    embedder=semantic_embedder, prepared_corpus=prepared_corpus,
                )
            if not isinstance(candidates, list) or len(candidates) > candidate_limit:
                raise ValueError("retriever exceeded the record candidate limit or returned invalid results")
            if any(not isinstance(record, Mapping) or not isinstance(record.get("research_id"), str)
                   or not record["research_id"].strip() or not isinstance(record.get("text"), str)
                   for record in candidates):
                raise ValueError("retrieval candidates require a non-empty research_id and full text string")
            details["candidate_ids"] = [record["research_id"] for record in candidates]
            if len(set(details["candidate_ids"])) != len(candidates):
                raise ValueError("retrieval candidate research_ids must be unique")
            for record in candidates:
                # Check full raw content and projected fields before truncation/JSON escaping.
                raw_values = [record["text"], record.get("title", ""), record.get("question", "")]
                raw_values.extend(record.get("sections", {}).values())
                raw_values.extend(record.get("metadata", {}).get(field) for field in ("date", "market", "status"))
                event = quarantine_untrusted_text(
                    "\n".join(value for value in raw_values if isinstance(value, str)),
                    source=f"retrieval:{record['research_id']}",
                )
                if event["status"] == "quarantined":
                    safety["events"].append(event)
                    details["quarantined_ids"].append(record["research_id"])
                    continue
                candidate = dict(record)
                candidate.pop("verification", None)  # A retriever cannot attest answer support.
                brief = _knowledge_planning_brief(candidate)
                safe_candidates.append(candidate)
                items.append({"id": candidate["research_id"], "kind": "retrieved_knowledge",
                              "text": json.dumps(brief, ensure_ascii=False, separators=(",", ":"))})
            retrieval_result = {
                "status": "ok" if safe_candidates else "abstain", "results": [],
                "rejected": [], "errors": [],
                "reason": "" if safe_candidates else "no safe retrieval candidates",
            }
        except Exception as exc:
            retrieval_result = {
                "status": "error", "results": [], "rejected": [],
                "errors": [{"error_type": "retrieval_error", "error": f"{type(exc).__name__}: {exc}"}],
                "reason": "retrieval failed closed",
            }
        details.setdefault("latency_ms", {})["runtime_total_ms"] = (perf_counter() - started) * 1000
        details["candidate_status"] = retrieval_result["status"]
        retrieval_result.update(details)
        retrieval_trace = {
            "status": retrieval_result["status"],
            "research_ids": [
                item["research_id"] for item in retrieval_result.get("results", [])
            ],
            **details,
            "reason": retrieval_result.get("reason", ""),
            "rejected": retrieval_result.get("rejected", []),
            "errors": retrieval_result.get("errors", []),
        }
        if retrieval_result["status"] == "error":
            error = (retrieval_result.get("errors") or [{}])[0]
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
        selected = select_context(items)["selected"]
        selected_ids = {item["id"] for item in selected if item["kind"] == "retrieved_knowledge"}
        safe_candidates = [record for record in safe_candidates if record["research_id"] in selected_ids]
        related_ids = [record["research_id"] for record in safe_candidates]
        retrieval_result["related_research_ids"] = retrieval_trace["related_research_ids"] = related_ids

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
            retrieval_result=retrieval_result,
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
        if plan["status"] == "no_action" and not _is_capability_request(user_request) and safe_candidates:
            tick = perf_counter()
            try:
                if retrieval_client is None:
                    raise ValueError("retrieval_client is required for knowledge-answer verification")
                verified = verify_candidates(
                    user_request, safe_candidates,
                    client=TelemetryClient(SafetyClient(retrieval_client), telemetry, stage="retrieval_verifier", model=model),
                    model=model,
                )
            except Exception as exc:
                verified = {"status": "error", "results": [], "rejected": [],
                            "errors": [{"error_type": "retrieval_error", "error": f"{type(exc).__name__}: {exc}"}],
                            "reason": "verification failed closed"}
            latency = retrieval_trace["latency_ms"]
            latency["verification_ms"] = (perf_counter() - tick) * 1000
            latency["runtime_total_ms"] += latency["verification_ms"]
            verified_ids = [record["research_id"] for record in verified["results"]]
            retrieval_result.update(verified)
            retrieval_result.update(verification_status=verified["status"], verified_ids=verified_ids)
            retrieval_trace.update({key: verified[key] for key in ("status", "reason", "rejected", "errors")})
            retrieval_trace.update(verification_status=verified["status"], verified_ids=verified_ids, research_ids=verified_ids)
            if verified["status"] == "error":
                error = verified["errors"][0]
                return _finish(
                    context_ids=[item["id"] for item in selected], planning=planning, steps=[],
                    retrieval=retrieval_trace, retrieval_result=retrieval_result,
                    research_run=None, grounding=None, hitl=None, plan=plan,
                    outcome="error", status="error", error_type=error["error_type"],
                    error_stage="retrieval", error=error["error"], safety=safety, telemetry=telemetry,
                )
            try:
                knowledge_evidence = _knowledge_records_to_evidence(verified["results"])
            except ValueError as exc:
                return _finish(
                    context_ids=[item["id"] for item in selected], planning=planning, steps=[],
                    retrieval=retrieval_trace, retrieval_result=retrieval_result,
                    research_run=None, grounding=None, hitl=None, plan=plan,
                    outcome="error", status="error", error_type="knowledge_evidence_limit",
                    error_stage="context", error=str(exc), safety=safety, telemetry=telemetry,
                )
            safe_knowledge = []
            for item in knowledge_evidence:
                snapshot = json.loads(item["text"])
                untrusted_text = "\n".join(value for value in (
                    snapshot["content"], snapshot["title"], *snapshot["provenance"].values(),
                    *snapshot["provenance"]["source_ref"],
                ) if isinstance(value, str))
                event = quarantine_untrusted_text(untrusted_text, source=f"knowledge_evidence:{item['id']}")
                if event["status"] == "quarantined":
                    safety["events"].append(event)
                else:
                    safe_knowledge.append(item)
            retrieval_trace["knowledge_evidence_ids"] = [item["id"] for item in safe_knowledge]
            if knowledge_evidence and not safe_knowledge:
                safety.update({"status": "blocked", "rule": "untrusted_instruction_injection",
                               "reason": "all knowledge evidence was quarantined"})
                return _finish(
                    context_ids=[item["id"] for item in selected], planning=planning, steps=[],
                    retrieval=retrieval_trace, retrieval_result=retrieval_result,
                    research_run=None, grounding=None, hitl=None, plan=plan,
                    outcome="blocked", safety=safety, telemetry=telemetry,
                )
            if safe_knowledge:
                return _synthesize_and_ground(
                    user_request, safe_knowledge, synthesis_client=synthesis_client,
                    grounding_client=grounding_client, model=model, telemetry=telemetry,
                    conversation_history=conversation_history,
                    context_ids=[item["id"] for item in selected], planning=planning, steps=[],
                    retrieval=retrieval_trace, retrieval_result=retrieval_result,
                    research_run=None, hitl=None, plan=plan, safety=safety,
                )
        if plan["status"] == "no_action" and _is_capability_request(user_request):
            answer, capability_evidence, grounding = _capability_answer()
            evidence_ids = [item["id"] for item in capability_evidence]
            return _finish(
                context_ids=[item["id"] for item in selected],
                planning=planning,
                steps=[],
                retrieval=retrieval_trace,
                research_run=None,
                grounding=grounding,
                grounding_trace={
                    "fully_grounded": True,
                    "labels": ["supported"] * len(evidence_ids),
                },
                hitl=None,
                outcome="success",
                plan=plan,
                retrieval_result=retrieval_result,
                answer=answer,
                evidence=capability_evidence,
                synthesis={
                    "status": "success",
                    "answer": answer,
                    "evidence_ids": evidence_ids,
                },
                safety=safety,
                telemetry=telemetry,
            )
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
            answer=(
                _clarification_answer(plan["reason"])
                if plan["status"] == "needs_input"
                else None
            ),
            safety=safety,
            telemetry=telemetry,
        )

    if hitl_client is None:
        raise ValueError("hitl_client is required for a ready plan")
    run = ResearchRun(
        user_request=user_request,
        **({"run_id": run_id} if run_id is not None else {}),
    )
    tool_results = []
    hitl_trace = None

    def gate_before_execute(_: dict[str, Any]) -> dict[str, Any]:
        nonlocal hitl_trace
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
            return {
                "status": "error",
                "error_type": gate["error_type"],
                "error_stage": "hitl",
                "error": gate["error"],
            }
        hitl_trace = {"decision": gate["decision"]}
        if gate["decision"] != "proceed":
            return {"status": "stop", "outcome": gate["decision"]}
        return {"status": "proceed"}

    def execute_agent_step(context: dict[str, Any]) -> ToolResult | None:
        step = context["step"]
        if step["name"] != "run_research_experiment":
            return None
        normalized_args = {"spec": step["arguments"]["spec"]}
        if experiment_authoring_client is None:
            return ToolResult.error(
                step["name"],
                normalized_args,
                "experiment_authoring_unavailable",
                "experiment_authoring_client is required",
                run_id=context["run"].run_id,
                provenance={"module": "agent.tools.experiment"},
            )

        authoring_client = TelemetryClient(
            SafetyClient(experiment_authoring_client),
            telemetry,
            stage="experiment_authoring",
            model=model,
        )
        feedback = None
        repairable_authoring = {"invalid_experiment_source", "malformed_response"}
        repairable_execution = {
            "experiment_import_error",
            "experiment_invalid_result",
            "experiment_runtime_error",
        }
        for attempt in range(2):
            authored = author_experiment(
                user_request,
                step["arguments"]["spec"],
                client=authoring_client,
                model=model,
                repair_feedback=feedback,
            )
            if authored["status"] == "error":
                if attempt == 0 and authored["error_type"] in repairable_authoring:
                    feedback = f"Source validation failed: {authored['error']}"
                    continue
                return ToolResult.error(
                    step["name"],
                    normalized_args,
                    authored["error_type"],
                    authored["error"],
                    run_id=context["run"].run_id,
                    provenance={"module": "agent.tools.experiment"},
                )
            executed = run_research_experiment(
                step["arguments"]["spec"],
                authored_program=authored["program"],
                authoring_provenance=authored["provenance"],
                run_id=context["run"].run_id,
            )
            executed.provenance["repair_attempts"] = attempt
            if executed.status != "error":
                return executed
            code = executed.errors[0]["code"]
            if attempt == 0 and code in repairable_execution:
                feedback = f"Execution failed with {code}: {executed.errors[0]['message']}"
                continue
            return executed
        raise AssertionError("bounded experiment repair loop exhausted")

    planner_decider = (
        loop_decider
        if loop_decider is not None
        else _planner_loop_decider(
            construction["input"], planner_client, telemetry, model
        )
    )
    try:
        loop_result = run_loop(
            user_request,
            decide_next=planner_decider,
            initial_steps=plan["steps"],
            run=run,
            tool_results=tool_results,
            before_execute=gate_before_execute,
            execute_step=execute_agent_step,
            max_iterations=max_iterations,
        )
    except Exception as exc:
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=_steps(run),
            retrieval=retrieval_trace,
            research_run=run if run.steps else None,
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

    run = loop_result["research_run"] or run
    visible_run = run if run.steps else None
    loop_trace = {
        "iterations": loop_result["iterations"],
        "outcome": loop_result["outcome"],
    }
    if loop_result["status"] == "error":
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=_steps(run),
            retrieval=retrieval_trace,
            research_run=visible_run,
            grounding=None,
            hitl=hitl_trace,
            outcome="error",
            plan=plan,
            retrieval_result=retrieval_result,
            status="error",
            error_type=loop_result["error_type"],
            error=loop_result["error"],
            error_stage=loop_result["error_stage"],
            safety=safety,
            telemetry=telemetry,
            loop=loop_trace,
        )

    steps = _steps(run)
    if run.status == "failed":
        failed_step = next((step for step in reversed(run.steps) if step["status"] == "error"), {})
        failure = (failed_step.get("errors") or [{
            "code": "research_run_failed",
            "message": "ResearchRun failed without a structured tool error.",
        }])[0]
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
            error_type=failure["code"],
            error=failure["message"],
            error_stage="execution",
            safety=safety,
            telemetry=telemetry,
            loop=loop_trace,
        )

    if loop_result["outcome"] in {
        "needs_input", "no_action", "needs_approval", "blocked"
    }:
        return _finish(
            context_ids=[item["id"] for item in selected],
            planning=planning,
            steps=steps,
            retrieval=retrieval_trace,
            research_run=visible_run,
            grounding=None,
            hitl=hitl_trace,
            outcome=loop_result["outcome"],
            plan=plan,
            retrieval_result=retrieval_result,
            answer=(
                _clarification_answer(
                    (loop_result.get("decision") or {}).get("reason")
                )
                if loop_result["outcome"] == "needs_input"
                else None
            ),
            safety=safety,
            telemetry=telemetry,
            loop=loop_trace,
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
    return _synthesize_and_ground(
        user_request, bound_evidence, synthesis_client=synthesis_client,
        grounding_client=grounding_client, model=model, telemetry=telemetry,
        conversation_history=conversation_history, answer=answer, evidence=evidence,
        context_ids=[item["id"] for item in selected],
        planning=planning,
        steps=steps,
        retrieval=retrieval_trace,
        research_run=run,
        hitl=hitl_trace,
        plan=plan,
        retrieval_result=retrieval_result,
        safety=safety,
        loop=loop_trace,
    )


__all__ = ["run_agent", "tool_results_to_evidence"]
