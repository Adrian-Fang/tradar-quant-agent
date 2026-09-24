"""Local-only HTTP adapter for the single-turn Tradar Agent runtime."""

from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.applications import Starlette
from starlette.routing import Route

from .main import run_request


_request_lock = Lock()


def _runtime_view(result: dict[str, Any]) -> dict[str, Any]:
    observed = result.get("observed") or {}
    research_run = result.get("research_run")
    if isinstance(research_run, dict):
        run_id = research_run.get("run_id")
    else:
        run_id = getattr(research_run, "run_id", None)
    outcome = observed.get("outcome") or {}
    telemetry = result.get("telemetry") or {}
    return {
        "status": result.get("status"),
        "outcome": outcome.get("status"),
        "answer": result.get("answer"),
        "run_id": run_id,
        "steps": observed.get("steps", []),
        "loop": observed.get("loop"),
        "grounding": observed.get("grounding"),
        "telemetry": telemetry.get("summary") if isinstance(telemetry, dict) else None,
        "error_type": result.get("error_type"),
        "error_stage": result.get("error_stage"),
        "error": result.get("error", ""),
    }


async def healthz(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def research(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except (TypeError, ValueError):
        return JSONResponse(
            {"status": "error", "error_type": "invalid_request", "error": "request body must be JSON"},
            status_code=400,
        )

    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        return JSONResponse(
            {"status": "error", "error_type": "invalid_request", "error": "message must be a non-empty string"},
            status_code=400,
        )

    if not _request_lock.acquire(blocking=False):
        return JSONResponse(
            {
                "status": "busy",
                "outcome": "busy",
                "answer": None,
                "run_id": None,
                "steps": [],
                "loop": None,
                "grounding": None,
                "telemetry": None,
                "error_type": "busy",
                "error_stage": "orchestration",
                "error": "another research request is already running",
            },
            status_code=503,
        )

    try:
        result = await asyncio.to_thread(
            run_request,
            message,
            provider="deepseek",
            retrieval="none",
        )
        return JSONResponse(_runtime_view(result))
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "outcome": "error",
                "answer": None,
                "run_id": None,
                "steps": [],
                "loop": None,
                "grounding": None,
                "telemetry": None,
                "error_type": "application_error",
                "error_stage": "orchestration",
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_code=500,
        )
    finally:
        _request_lock.release()


app = Starlette(
    routes=[
        Route("/healthz", healthz, methods=["GET"]),
        Route("/v1/research", research, methods=["POST"]),
    ]
)


__all__ = ["app", "healthz", "research"]
