"""Model-first Agent: safety, native tool loop, then evidence-backed answers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from .answer.synthesizer import synthesize_answer
from .core.contracts import ResearchRun, ToolResult
from .core.providers import ProviderError, normalize_response_output, provider_error_type
from .core.resources import load_json, load_prompt
from .core.safety import SafetyClient, check_request_safety, quarantine_untrusted_text
from .core.telemetry import RunTelemetry, TelemetryClient
from .grounding.verifier import verify_answer_grounding
from .hitl.gate import gate_action
from .retrieval.hybrid_retriever import KnowledgeRetriever
from .retrieval.qdrant_store import filter_spec
from .retrieval.relevance_verifier import RECORD_EVIDENCE_FIELDS, verify_candidates
from .retrieval.semantic_retriever import retrieve_semantic
from .tools.calling import TOOL_SCHEMAS, _extract_function_calls, normalize_tool_arguments
from .tools.executor import execute_steps
from .tools.experiment import author_experiment, run_research_experiment

CAPABILITY_ITEMS = load_json("capabilities.json")
PROMPT = load_prompt("prompts/agent.md")
KNOWLEDGE_RECORD_LIMIT = 5
KNOWLEDGE_RECORD_CHARS = 12000
RELATED_CONTEXT_CHARS = 6000
LOOKUP_QUERY_CHARS = 1000
RESOLVED_QUERY_CONTEXT_CHARS = 2000
ANSWER_TARGET_CHARS = 2000
READ_ONLY_RESEARCH_TOOLS = frozenset({
    "inspect_universe", "evaluate_factor", "run_backtest", "run_research_experiment",
})
UTILITY_SCHEMAS = (
    {"type": "function", "name": "request_clarification",
     "description": "Ask the minimum question needed when research intent is materially incomplete.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                    "required": ["question"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "search_knowledge",
     "description": f"Find canonical historical research using a self-contained query (at most {LOOKUP_QUERY_CHARS} characters) that resolves references from safe history. Returns related context, NOT verified answer evidence. Use for historical questions or method reuse; not for generic chat.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"},
         "answer_target": {"type": ["string", "null"], "maxLength": ANSWER_TARGET_CHARS,
                           "description": "Explicit selection of ONE self-contained historical question being answered, resolved from the current user turn and bounded safe conversation context. Preserve the user's subject/date/scope; do not include exploratory search objectives. Send null to keep the current answer target (initially the exact user turn)."}},
                    "required": ["query", "answer_target"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "lookup_history",
     "description": "Read a bounded older-only slice omitted from initial context, once per request. Use when recent history is insufficient.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True},
)


def _split_history(history: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Recent slice and its exact complement, including omitted message prefixes."""
    recent, older, remaining = [], [dict(message) for message in history], 2000
    for message in reversed(history[-4:]):
        if remaining == 0:
            break
        content = message["content"][-remaining:]
        recent.append({"role": message["role"], "content": content})
        older.pop()
        if len(content) < len(message["content"]):
            older.append({"role": message["role"], "content": message["content"][:-len(content)]})
        remaining -= len(content)
    return list(reversed(recent)), older


def _execute_experiment(user_request, spec, run_id, client, telemetry, model):
    """Keep authoring/isolated execution and the existing one-repair contract."""
    if client is None:
        return ToolResult.error("run_research_experiment", {"spec": spec},
                                "experiment_authoring_unavailable", "experiment_authoring_client is required",
                                run_id=run_id)
    authoring_client = TelemetryClient(SafetyClient(client), telemetry, stage="experiment_authoring", model=model)
    feedback = None
    for attempt in range(2):
        authored = author_experiment(user_request, spec, client=authoring_client, model=model,
                                     repair_feedback=feedback)
        if authored["status"] == "error":
            if attempt == 0 and authored["error_type"] in {"invalid_experiment_source", "malformed_response"}:
                feedback = f"Source validation failed: {authored['error']}"
                continue
            return ToolResult.error("run_research_experiment", {"spec": spec}, authored["error_type"],
                                    authored["error"], run_id=run_id,
                                    provenance={"module": "agent.tools.experiment"})
        result = run_research_experiment(spec, authored_program=authored["program"],
                                        authoring_provenance=authored["provenance"], run_id=run_id)
        result.provenance["repair_attempts"] = attempt
        if result.status != "error":
            return result
        code = result.errors[0]["code"]
        if attempt == 0 and code in {"experiment_import_error", "experiment_invalid_result", "experiment_runtime_error"}:
            feedback = f"Execution failed with {code}: {result.errors[0]['message']}"
            continue
        return result
    raise AssertionError("bounded experiment repair loop exhausted")


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


def _related_research_brief(record: dict[str, Any]) -> dict[str, Any]:
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
        if len(text) > KNOWLEDGE_RECORD_CHARS:
            raise ValueError(f"knowledge record {research_id} exceeds the {KNOWLEDGE_RECORD_CHARS}-character evidence limit")
        evidence.append({"id": f"knowledge-{research_id}", "text": text})
        if len(evidence) == KNOWLEDGE_RECORD_LIMIT:
            break
    return evidence


def _finish(
    *,
    context_ids: list[str],
    steps: list[dict[str, Any]],
    retrieval: dict[str, Any],
    research_run: ResearchRun | None,
    grounding: dict[str, Any] | None,
    grounding_trace: dict[str, Any] | None = None,
    hitl: dict[str, Any] | None,
    outcome: str,
    retrieval_result: dict[str, Any] | None = None,
    status: str = "ok",
    error_type: str | None = None,
    error: str = "",
    error_stage: str | None = None,
    answer: str | None = None,
    evidence: list[dict[str, str]] | None = None,
    synthesis: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
    telemetry: RunTelemetry | None = None,
    loop: dict[str, Any] | None = None,
    model_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if telemetry is not None:
        runtime_stage = error_stage
        if runtime_stage is None and outcome == "needs_input":
            runtime_stage = "model"
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
        # Deprecated null fields retained solely for CLI/JSON consumers.
        "planning": None,
        "steps": steps,
        "retrieval": retrieval,
        "research_run": (
            {"status": research_run.status, "final_status": research_run.final_status}
            if research_run is not None else None
        ),
        "grounding": grounding_trace if grounding_trace is not None else grounding,
        "hitl": hitl,
        "orchestration": None,
        "outcome": {"status": outcome},
    }
    if loop is not None:
        observed["loop"] = {**loop, "outcome": outcome}
    if model_trace is not None:
        observed["model"] = {"tool_calls": model_trace}
    return {
        "status": status,
        "plan": None,
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
    user_request: str, *, client: Any | None = None, planner_client: Any | None = None,
    experiment_authoring_client: Any | None = None, hitl_client: Any | None = None,
    synthesis_client: Any | None = None, grounding_client: Any | None = None,
    retrieval_client: Any | None = None, retrieval_backend: str = "legacy",
    retrieval_strategy: str = "dense", knowledge_retriever: Any | None = None,
    retrieval_filters: dict[str, Any] | None = None, semantic_embedder: Any | None = None,
    prepared_corpus: dict[str, Any] | None = None, context_items: list[dict[str, Any]] | None = None,
    proposed_action: dict[str, Any] | None = None, existing_approval: dict[str, Any] | None = None,
    product_boundaries: list[str] | None = None, model: str = "", run_id: str | None = None,
    candidate_limit: int = 5, market: str | None = None, topic: str | None = None,
    record_status: str | None = None, max_iterations: int = 8,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """One model owns direct answers, clarification, lookup and research actions.

    Deprecated: planner_client aliases client for existing callers; it does not
    create a planner stage. New callers should use client.
    """
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("user_request must be a non-empty string")
    if client is not None and planner_client is not None and client is not planner_client:
        raise ValueError("supply client or planner_client, not two different primary clients")
    client = client if client is not None else planner_client
    if client is None:
        raise ValueError("client is required")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    telemetry = RunTelemetry()
    boundaries = product_boundaries or []
    safety = check_request_safety(user_request, proposed_action or {}, boundaries)
    run, results, calls, history = None, [], [], []
    retrieval_result, hitl_trace = None, None
    retrieval_trace = {"status": "not_used", "candidate_status": "not_used", "research_ids": []}
    # Ordered, deduplicated related context; never evidence before verification.
    # The global pool and verifier input share the final evidence record cap.
    safe_candidates = {}
    lookup_queries = []
    historical_answer_target = user_request
    knowledge_observation = None
    history_lookup_used = False
    context_ids = ["request_scope"]

    def finish(outcome, **kwargs):
        if run is not None and run.status == "running":
            if outcome == "error":
                run.fail()
            else:
                run.complete(final_status="partial")
        return _finish(context_ids=context_ids, steps=_steps(run) if run else [],
                       retrieval=retrieval_trace, retrieval_result=retrieval_result, research_run=run,
                       grounding=None, hitl=hitl_trace, outcome=outcome, safety=safety,
                       telemetry=telemetry, model_trace=calls,
                       loop={"iterations": len(calls)} if calls else None, **kwargs)

    def error(code, message, stage):
        return finish("error", status="error", error_type=code, error=message, error_stage=stage)

    if safety["status"] == "blocked":
        return finish("blocked")
    if retrieval_backend not in {"legacy", "qdrant"} or retrieval_strategy not in {"dense", "hybrid"}:
        raise ValueError("invalid retrieval backend or strategy")
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= KNOWLEDGE_RECORD_LIMIT:
        raise ValueError("candidate_limit must be between 1 and 5")
    if retrieval_backend == "legacy" and (knowledge_retriever is not None or retrieval_filters):
        raise ValueError("knowledge_retriever and retrieval_filters require the qdrant backend")
    filters = dict(retrieval_filters or {})
    for key, value in (("market", market), ("topic", topic), ("status", record_status)):
        if value is not None:
            if key in filters and filters[key] != value:
                raise ValueError(f"conflicting retrieval filter: {key}")
            filters[key] = value
    if retrieval_backend == "qdrant":
        filter_spec(filters)

    def safe_text(text, source):
        event = quarantine_untrusted_text(text, source=source)
        if event["status"] == "quarantined":
            safety["events"].append(event)
            return False
        return True

    for index, message in enumerate(conversation_history or []):
        if safe_text(message["content"], f"history:{index}"):
            history.append(dict(message))
    recent, older = _split_history(history)
    older_lookup, omitted_older = _split_history(older)
    messages = [*recent, {"role": "user", "content": user_request}]
    for item in context_items or []:
        if safe_text(item.get("text", ""), f"context:{item['id']}"):
            messages.append({"role": "user", "content": "Unverified related context: " + item["text"][:2000]})
            context_ids.append(item["id"])
    schemas = [*TOOL_SCHEMAS, *(schema for schema in UTILITY_SCHEMAS
                if schema["name"] != "search_knowledge" or retrieval_backend == "qdrant" or semantic_embedder is not None)]
    schema_by_name = {schema["name"]: schema for schema in schemas}
    primary = TelemetryClient(SafetyClient(client), telemetry, stage="model", model=model)

    for iteration in range(max_iterations + 1):
        try:
            response = primary.create({
                "model": model, "instructions": PROMPT + "\nRuntime metadata: " + json.dumps(
                    {"model": model or None, "capabilities": CAPABILITY_ITEMS, "boundaries": boundaries},
                    ensure_ascii=False),
                "input": messages, "tools": schemas, "tool_choice": "auto", "parallel_tool_calls": False,
            })
        except Exception as exc:
            return error(provider_error_type(exc), f"{type(exc).__name__}: {exc}", "model")
        try:
            raw_output = response.get("output", []) if isinstance(response, Mapping) else getattr(response, "output", [])
            output_items = normalize_response_output(raw_output)
            selected = _extract_function_calls({"output": output_items})
            text = response.get("output_text", "") if isinstance(response, Mapping) else getattr(response, "output_text", "")
            if len(selected) > 1:
                raise ValueError("select at most one tool per turn")
            if not selected and (not isinstance(text, str) or not text.strip()):
                raise ValueError("model must return text or one tool call")
            if selected:
                call = selected[0]
                name, arguments = call["name"], call["arguments"]
                if name not in schema_by_name:
                    raise ValueError(f"unsupported model tool: {name}")
                parameters = schema_by_name[name]["parameters"]
                missing = set(parameters.get("required", [])) - set(arguments)
                # Legacy/non-strict callers may omit fixed-tool default fields.
                if name in READ_ONLY_RESEARCH_TOOLS and schema_by_name[name]["strict"]:
                    missing -= {key for key, prop in parameters["properties"].items() if "null" in prop.get("type", [])}
                if set(arguments) - set(parameters["properties"]) or missing:
                    raise ValueError(f"invalid arguments for {name}")
                arguments = normalize_tool_arguments(name, arguments)
        except ProviderError as exc:
            return error(exc.code, str(exc), "model")
        except (ValueError, TypeError) as exc:
            return error("malformed_response", str(exc), "model")

        if not selected:
            if run is not None:
                run.complete(final_status="partial" if any(item.status == "partial" for item in results) else "success")
                bound_evidence = tool_results_to_evidence(results)
            elif retrieval_result is not None:
                started = perf_counter()
                # Search objectives are not constraints on the question being answered.
                verification_query = historical_answer_target
                retrieval_trace["verification_query"] = verification_query
                verified = verify_candidates(
                    verification_query, list(safe_candidates.values()),
                    client=TelemetryClient(SafetyClient(retrieval_client or client), telemetry, stage="retrieval_verifier", model=model),
                    model=model,
                )
                retrieval_result.update(verified)
                retrieval_trace.update(status=verified["status"], verification_status=verified["status"],
                                       verified_ids=[item["research_id"] for item in verified["results"]],
                                       research_ids=[item["research_id"] for item in verified["results"]],
                                       errors=verified["errors"], rejected=verified["rejected"])
                retrieval_trace.setdefault("latency_ms", {})["verification_ms"] = (perf_counter() - started) * 1000
                retrieval_trace["runtime_total_ms"] = (
                    retrieval_trace["latency_ms"]["candidate_ms"] + retrieval_trace["latency_ms"]["verification_ms"]
                )
                if verified["status"] == "error":
                    failure = verified["errors"][0]
                    return error(failure["error_type"], failure["error"], "retrieval")
                try:
                    bound_evidence = _knowledge_records_to_evidence(verified["results"])
                except (ValueError, TypeError) as exc:
                    return error("knowledge_evidence_error", str(exc), "retrieval")
            else:
                return finish("success", answer=text.strip(), evidence=[])
            if not bound_evidence:
                return finish("abstain", answer="没有足够的已验证 evidence 来回答这个问题。", evidence=[])
            return _synthesize_and_ground(
                user_request, bound_evidence, synthesis_client=synthesis_client or client,
                grounding_client=grounding_client or client, model=model, telemetry=telemetry,
                conversation_history=history, context_ids=context_ids,
                steps=_steps(run) if run else [], retrieval=retrieval_trace, retrieval_result=retrieval_result,
                research_run=run, hitl=hitl_trace, safety=safety, model_trace=calls,
                loop={"iterations": len(calls)},
            )

        if iteration == max_iterations:
            if run is not None:
                run.fail()
            return error("max_iterations_exceeded", "Agent tool-call budget exhausted.", "loop")
        calls.append({"name": name, "arguments": arguments})
        if name == "request_clarification":
            question = arguments["question"]
            if not isinstance(question, str) or not question.strip():
                return error("malformed_response", "clarification.question must be a non-empty string", "model")
            if run is not None:
                run.complete(final_status="partial")
            return finish("needs_input", answer=question.strip(), evidence=[])
        call_id = call.get("call_id") or f"call-{iteration + 1}"
        # Replay Responses output verbatim (reasoning, messages, and the call), once.
        messages.extend(output_items)
        if name == "lookup_history":
            observation = {"history": [] if history_lookup_used else older_lookup,
                           "already_read": history_lookup_used, "truncated": bool(omitted_older)}
            history_lookup_used = True
        elif name == "search_knowledge":
            query = arguments["query"]
            if not isinstance(query, str) or not query.strip():
                return error("malformed_response", "search_knowledge.query must be a non-empty string", "model")
            if len(query) > LOOKUP_QUERY_CHARS:
                return error("malformed_response", f"search_knowledge.query exceeds {LOOKUP_QUERY_CHARS} characters", "model")
            if arguments["answer_target"] is not None:
                target = arguments["answer_target"]
                if not isinstance(target, str) or not target.strip() or len(target) > ANSWER_TARGET_CHARS:
                    return error("malformed_response", f"search_knowledge.answer_target must be a non-empty string of at most {ANSWER_TARGET_CHARS} characters", "model")
                historical_answer_target = target.strip()
            query = query.strip()
            if query not in lookup_queries:
                lookup_queries.append(query)
            while len("\n".join(lookup_queries)) > RESOLVED_QUERY_CONTEXT_CHARS:
                lookup_queries.pop(0)
            started = perf_counter()
            try:
                if retrieval_backend == "qdrant":
                    retriever = knowledge_retriever if knowledge_retriever is not None else KnowledgeRetriever()
                    search = retriever.search(query, mode=retrieval_strategy, limit=candidate_limit, **filters)
                    candidates = search["results"]
                    retrieval_trace.update(backend="qdrant", mode=retrieval_strategy, filters=filters,
                                           chunks_returned=search.get("chunks_returned"))
                    latency = retrieval_trace.setdefault("latency_ms", {})
                    for key, value in search.get("latency_ms", {}).items():
                        latency[key] = latency.get(key, 0) + value
                else:
                    candidates = retrieve_semantic(query, limit=candidate_limit, market=market, topic=topic,
                                                   status=record_status, embedder=semantic_embedder, prepared_corpus=prepared_corpus)
                    retrieval_trace.update(backend="legacy")
                if not isinstance(candidates, list) or len(candidates) > candidate_limit:
                    raise ValueError("retriever exceeded candidate limit or returned invalid results")
                ids = [record["research_id"] for record in candidates]
                if any(not isinstance(identity, str) or not identity.strip() for identity in ids) or len(set(ids)) != len(ids):
                    raise ValueError("retrieval candidates require unique non-empty research_ids")
                quarantined = []
                for record in candidates:
                    if not isinstance(record.get("text"), str):
                        raise ValueError("retrieval candidate requires full text")
                    # Inspect the entire record before bounded projection, including provenance.
                    if not safe_text(json.dumps(record, ensure_ascii=False, default=str), f"retrieval:{record['research_id']}"):
                        quarantined.append(record["research_id"])
                        safe_candidates.pop(record["research_id"], None)
                        continue
                    candidate = dict(record)
                    candidate.pop("verification", None)
                    # Full verifier records are bounded without truncating research facts.
                    record_text = json.dumps({key: candidate[key] for key in RECORD_EVIDENCE_FIELDS if key in candidate},
                                             ensure_ascii=False)
                    if len(record_text) > KNOWLEDGE_RECORD_CHARS:
                        raise ValueError(f"retrieval candidate {record['research_id']} exceeds the {KNOWLEDGE_RECORD_CHARS}-character verifier record limit")
                    safe_candidates[record["research_id"]] = candidate
                evicted = []
                while True:
                    observation = {"context_type": "related_research_context",
                                   "records": [_related_research_brief(record) for record in safe_candidates.values()]}
                    if len(safe_candidates) <= candidate_limit and len(json.dumps(observation, ensure_ascii=False, sort_keys=True)) <= RELATED_CONTEXT_CHARS:
                        break
                    # Recent lookups displace oldest retained records; no score re-ranking.
                    identity = next(iter(safe_candidates))
                    safe_candidates.pop(identity)
                    evicted.append(identity)
                status = "ok" if safe_candidates else "abstain"
                retrieval_result = {"status": status, "results": [], "errors": [], "rejected": [],
                                    "candidate_ids": list(dict.fromkeys([*retrieval_trace.get("candidate_ids", []), *ids])),
                                    "related_research_ids": list(safe_candidates)}
                retrieval_trace.update(retrieval_result, candidate_status=status, research_ids=[],
                                       quarantined_ids=list(dict.fromkeys([*retrieval_trace.get("quarantined_ids", []), *quarantined])),
                                       verified_ids=[], verification_status="not_used", candidate_limit=candidate_limit)
                retrieval_trace.update(evicted_ids=evicted, resolved_lookup_queries=list(lookup_queries),
                                       related_context_char_limit=RELATED_CONTEXT_CHARS,
                                       verifier_record_char_limit=KNOWLEDGE_RECORD_CHARS)
            except Exception as exc:
                failure = {"error_type": "retrieval_error", "error": f"{type(exc).__name__}: {exc}"}
                retrieval_result = {**(retrieval_result or {}), "status": "error", "results": [], "errors": [failure], "rejected": []}
                retrieval_trace.update(status="error", candidate_status="error", errors=[failure], research_ids=[])
                return error("retrieval_error", f"{type(exc).__name__}: {exc}", "retrieval")
            finally:
                latency = retrieval_trace.setdefault("latency_ms", {})
                latency["candidate_ms"] = latency.get("candidate_ms", 0) + (perf_counter() - started) * 1000
                retrieval_trace["runtime_total_ms"] = latency["candidate_ms"] + latency.get("verification_ms", 0)
        else:
            # Current canonical tools are local/read-only; no LLM approval round-trip.
            if name in READ_ONLY_RESEARCH_TOOLS:
                gate = {"status": "ok", "decision": "proceed", "approval_request": None,
                        "reason": "Deterministic proceed for canonical local read-only research."}
            else:
                concrete_action = {**(proposed_action or {}), "tool": name, "arguments": arguments,
                                   "description": schema_by_name[name]["description"]}
                gate = gate_action(
                    user_request, concrete_action, existing_approval or {"approved": False, "scope": None},
                    boundaries, client=TelemetryClient(SafetyClient(hitl_client or client), telemetry, stage="hitl", model=model), model=model,
                )
            hitl_trace = {key: gate[key] for key in ("decision", "approval_request", "reason")}
            hitl_trace["mode"] = "deterministic_read_only" if name in READ_ONLY_RESEARCH_TOOLS else "approval_gate"
            if gate["status"] == "error":
                return error(gate["error_type"], gate["error"], "hitl")
            if gate["decision"] != "proceed":
                if run is not None:
                    run.complete(final_status="partial")
                return finish(gate["decision"], answer=gate["approval_request"], evidence=[])
            if run is None:
                run = ResearchRun(user_request=user_request, **({"run_id": run_id} if run_id is not None else {}))
            if name == "run_research_experiment":
                try:
                    result = _execute_experiment(user_request, arguments["spec"], run.run_id,
                                                 experiment_authoring_client or client, telemetry, model)
                except Exception as exc:
                    result = ToolResult.error(name, arguments, "step_execution_error", str(exc) or type(exc).__name__, run_id=run.run_id)
                run.add_step(result)
                results.append(result)
                if result.status == "error":
                    run.fail()
            else:
                execute_steps([{"name": name, "arguments": arguments}], run=run, tool_results=results, finalize=False)
                result = results[-1]
            if run.status == "failed":
                failure = (result.errors or [{"code": "research_run_failed", "message": "ResearchRun failed without a structured tool error."}])[0]
                return error(failure["code"], failure["message"], "execution")
            observation = json.loads(tool_results_to_evidence([result])[0]["text"])
            if not safe_text(json.dumps(observation, ensure_ascii=False), f"tool:{name}"):
                return finish("blocked", evidence=[])
        observation_message = {"type": "function_call_output", "call_id": call_id,
                               "output": json.dumps(observation, ensure_ascii=False, sort_keys=True)}
        if name == "search_knowledge":
            if knowledge_observation is not None:
                knowledge_observation["output"] = '{"superseded": true}'
            knowledge_observation = observation_message
        messages.append(observation_message)

    raise AssertionError("bounded model loop exhausted")


__all__ = ["run_agent", "tool_results_to_evidence"]
