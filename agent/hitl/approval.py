"""Deterministic, scope-bound HITL approval lifecycle."""

from __future__ import annotations

from typing import Any


def create_approval_request(
    gate_result: dict[str, Any],
    proposed_action: dict[str, Any],
    *,
    approval_id: str,
) -> dict[str, Any]:
    if not isinstance(gate_result, dict) or gate_result.get("status") != "ok":
        raise ValueError("gate_result must be a successful gate result")
    if gate_result.get("decision") != "needs_approval":
        raise ValueError("approval request requires needs_approval")
    approval_request = gate_result.get("approval_request")
    if not isinstance(approval_request, str) or not approval_request.strip():
        raise ValueError("needs_approval requires a non-empty approval_request")
    if not isinstance(approval_id, str) or not approval_id.strip():
        raise ValueError("approval_id must be non-empty")
    if not isinstance(proposed_action, dict) or set(proposed_action) != {
        "description", "environment", "reversible",
    }:
        raise ValueError("proposed_action must contain description, environment, and reversible")

    return {
        "approval_id": approval_id,
        "status": "pending",
        "action_snapshot": dict(proposed_action),
        "approval_request": approval_request,
    }


def resolve_approval(record: dict[str, Any], decision: str) -> dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    if not isinstance(record, dict) or record.get("status") != "pending":
        raise ValueError("only pending approval records can be resolved")
    if not record.get("approval_id") or not isinstance(record.get("action_snapshot"), dict):
        raise ValueError("record is missing approval provenance")

    resolved = dict(record)
    resolved["status"] = decision
    return resolved


def authorization_from_approval(
    record: dict[str, Any],
    proposed_action: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("status") not in {
        "pending", "approved", "rejected",
    }:
        raise ValueError("record has an invalid approval status")
    if not record.get("approval_id") or not isinstance(record.get("action_snapshot"), dict):
        raise ValueError("record is missing approval provenance")
    if not isinstance(proposed_action, dict):
        raise ValueError("proposed_action must be an object")

    authorized = (
        record["status"] == "approved"
        and proposed_action == record["action_snapshot"]
    )
    return {
        "approved": authorized,
        "scope": record["approval_request"] if authorized else None,
    }


__all__ = [
    "authorization_from_approval",
    "create_approval_request",
    "resolve_approval",
]
