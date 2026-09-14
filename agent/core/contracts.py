"""Small serializable contracts for Python-level research tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd


ToolStatus = Literal["success", "partial", "error"]
RunLifecycle = Literal["running", "completed", "failed"]


def _new_run_id() -> str:
    return uuid4().hex


def _json_ready(value: Any) -> Any:
    """Convert common research values without embedding frame contents."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe_summary",
            "rows": int(value.shape[0]),
            "columns": [str(column) for column in value.columns],
        }
    if isinstance(value, pd.Series):
        return {
            "type": "series_summary",
            "rows": int(value.shape[0]),
            "name": str(value.name) if value.name is not None else None,
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_ready(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _issues(items: list[Mapping[str, Any]] | None, default_code: str) -> list[dict[str, Any]]:
    normalized = []
    for item in items or []:
        if isinstance(item, str):
            normalized.append({"code": default_code, "message": item})
            continue
        if not isinstance(item, Mapping):
            raise TypeError(f"issue must be a mapping or string, got {type(item).__name__}")
        message = item.get("message")
        if message is None or str(message).strip() == "":
            raise ValueError("issue.message cannot be empty")
        normalized.append({
            "code": str(item.get("code", default_code)),
            "message": str(message),
            **{
                str(key): _json_ready(value)
                for key, value in item.items()
                if key not in {"code", "message"}
            },
        })
    return normalized


@dataclass
class ToolResult:
    """Serializable result envelope shared by Python-level research tools."""

    tool_name: str
    run_id: str = field(default_factory=_new_run_id)
    status: ToolStatus = "success"
    normalized_args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=lambda: {"elapsed_ms": 0.0})

    def __post_init__(self) -> None:
        if self.status not in {"success", "partial", "error"}:
            raise ValueError(f"unsupported ToolResult status: {self.status}")
        if not isinstance(self.normalized_args, Mapping):
            raise TypeError("normalized_args must be a mapping")
        if not isinstance(self.timing, Mapping):
            raise TypeError("timing must be a mapping")
        self.normalized_args = dict(self.normalized_args)
        self.warnings = _issues(self.warnings, "warning")
        self.errors = _issues(self.errors, "error")
        self.artifacts = list(self.artifacts or [])
        self.provenance = dict(self.provenance or {})
        self.timing = dict(self.timing)
        self.timing.setdefault("elapsed_ms", 0.0)

    @classmethod
    def error(
        cls,
        tool_name: str,
        normalized_args: Mapping[str, Any],
        code: str,
        message: str,
        *,
        run_id: str | None = None,
        warnings: list[dict[str, Any]] | None = None,
        provenance: Mapping[str, Any] | None = None,
        elapsed_ms: float = 0.0,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            run_id=run_id or _new_run_id(),
            status="error",
            normalized_args=dict(normalized_args),
            warnings=warnings or [],
            errors=[{"code": code, "message": message}],
            provenance=dict(provenance or {}),
            timing={"elapsed_ms": round(float(elapsed_ms), 3)},
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready({
            "tool_name": self.tool_name,
            "run_id": self.run_id,
            "status": self.status,
            "normalized_args": self.normalized_args,
            "result": self.result,
            "warnings": self.warnings,
            "errors": self.errors,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
            "timing": self.timing,
        })

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolResult":
        return cls(
            tool_name=str(payload["tool_name"]),
            run_id=str(payload["run_id"]),
            status=payload["status"],
            normalized_args=dict(payload.get("normalized_args", {})),
            result=payload.get("result"),
            warnings=list(payload.get("warnings", [])),
            errors=list(payload.get("errors", [])),
            artifacts=list(payload.get("artifacts", [])),
            provenance=dict(payload.get("provenance", {})),
            timing=dict(payload.get("timing", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ToolResult":
        return cls.from_dict(json.loads(payload))


def _result_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {"type": "mapping", "keys": sorted(str(key) for key in value)}
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "rows": int(value.shape[0]),
            "columns": [str(c) for c in value.columns],
        }
    return {"type": type(value).__name__}


@dataclass
class ResearchRun:
    """Minimal ordered trace for replay/evaluation of one research request."""

    run_id: str = field(default_factory=_new_run_id)
    user_request: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    validation: Any = None
    final_status: str = "running"
    human_intervention: list[dict[str, Any]] = field(default_factory=list)
    status: RunLifecycle = "running"

    def __post_init__(self) -> None:
        if self.status not in {"running", "completed", "failed"}:
            raise ValueError(f"unsupported ResearchRun status: {self.status}")
        allowed_final_status = {
            "running": {"running"},
            "completed": {"success", "partial"},
            "failed": {"error"},
        }
        if self.final_status not in {"running", "success", "partial", "error"}:
            raise ValueError(f"unsupported ResearchRun final_status: {self.final_status}")
        if self.final_status not in allowed_final_status[self.status]:
            raise ValueError(
                f"ResearchRun status {self.status} conflicts with final_status {self.final_status}"
            )

    def complete(self, *, final_status: Literal["success", "partial"] = "success") -> None:
        if final_status not in {"success", "partial"}:
            raise ValueError(f"unsupported completed final_status: {final_status}")
        if self.status != "running":
            raise RuntimeError(f"cannot complete ResearchRun from status: {self.status}")
        self.status = "completed"
        self.final_status = final_status

    def fail(self) -> None:
        if self.status != "running":
            raise RuntimeError(f"cannot fail ResearchRun from status: {self.status}")
        self.status = "failed"
        self.final_status = "error"

    def add_step(
        self,
        tool_result: ToolResult,
        *,
        result_summary: Mapping[str, Any] | None = None,
        result_ref: Any = None,
    ) -> dict[str, Any]:
        if self.status != "running":
            raise RuntimeError(f"cannot add step to ResearchRun from status: {self.status}")
        if not isinstance(tool_result, ToolResult):
            raise TypeError("ResearchRun steps require a ToolResult")
        step = {
            "seq": len(self.steps) + 1,
            "tool_name": tool_result.tool_name,
            "normalized_args": tool_result.normalized_args,
            "status": tool_result.status,
            "warnings": tool_result.warnings,
            "errors": tool_result.errors,
            "artifacts": tool_result.artifacts,
        }
        if result_ref is not None:
            step["result_ref"] = result_ref
        else:
            step["result_summary"] = dict(result_summary or _result_summary(tool_result.result))
        self.steps.append(step)
        return step

    def to_dict(self) -> dict[str, Any]:
        return _json_ready({
            "run_id": self.run_id,
            "user_request": self.user_request,
            "steps": self.steps,
            "validation": self.validation,
            "final_status": self.final_status,
            "human_intervention": self.human_intervention,
            "status": self.status,
        })

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchRun":
        final_status = str(payload.get("final_status", "running"))
        if "status" in payload:
            status = payload["status"]
        else:
            status = {
                "running": "running",
                "success": "completed",
                "partial": "completed",
                "error": "failed",
            }.get(final_status)
            if status is None:
                raise ValueError(f"unsupported ResearchRun final_status: {final_status}")
        return cls(
            run_id=str(payload["run_id"]),
            user_request=str(payload.get("user_request", "")),
            steps=list(payload.get("steps", [])),
            validation=payload.get("validation"),
            final_status=final_status,
            human_intervention=list(payload.get("human_intervention", [])),
            status=status,
        )

    @classmethod
    def from_json(cls, payload: str) -> "ResearchRun":
        return cls.from_dict(json.loads(payload))


__all__ = ["ResearchRun", "RunLifecycle", "ToolResult", "ToolStatus"]
