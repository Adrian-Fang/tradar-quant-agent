"""Small provider-call telemetry contract for one Agent run."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

from .safety import TRUST_BOUNDARY_INSTRUCTIONS


# Prices are explicit configuration, separate from provider usage. Values are
# USD per million tokens; callers should update this table when billing changes.
MODEL_PRICING = {
    ("deepseek", "deepseek-v4-flash"): {
        "input_per_million": 0.14,
        "output_per_million": 0.28,
    },
    ("openai", "gpt-5"): {
        "input_per_million": 1.25,
        "output_per_million": 10.0,
    },
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _usage(response: Any) -> dict[str, int | None]:
    usage = _field(response, "usage")
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
        }

    prompt_details = _field(usage, "prompt_tokens_details") or _field(
        usage, "input_token_details"
    )
    completion_details = _field(usage, "completion_tokens_details") or _field(
        usage, "output_token_details"
    )
    return {
        "input_tokens": _token_count(
            _field(usage, "input_tokens", _field(usage, "prompt_tokens"))
        ),
        "output_tokens": _token_count(
            _field(usage, "output_tokens", _field(usage, "completion_tokens"))
        ),
        "cached_tokens": _token_count(
            _field(usage, "cached_tokens", _field(prompt_details, "cached_tokens"))
        ),
        "reasoning_tokens": _token_count(
            _field(usage, "reasoning_tokens", _field(completion_details, "reasoning_tokens"))
        ),
    }


def _provider_name(client: Any) -> str | None:
    provider = getattr(client, "provider", None)
    if isinstance(provider, str) and provider:
        return provider
    names = {
        "DeepSeekChatClient": "deepseek",
        "OpenAIResponsesClient": "openai",
        "FixtureClient": "fixture",
    }
    return names.get(type(client).__name__)


def estimate_cost(
    provider: str | None,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if provider is None or model is None or input_tokens is None or output_tokens is None:
        return None
    pricing = MODEL_PRICING.get((provider, model))
    if pricing is None:
        return None
    return round(
        input_tokens * pricing["input_per_million"] / 1_000_000
        + output_tokens * pricing["output_per_million"] / 1_000_000,
        8,
    )


def _aggregate(calls: list[dict[str, Any]]) -> dict[str, Any]:
    def total(field: str) -> int | float | None:
        values = [call[field] for call in calls]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    return {
        "calls": len(calls),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "cached_tokens": total("cached_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "total_tokens": (
            total("input_tokens") + total("output_tokens")
            if total("input_tokens") is not None and total("output_tokens") is not None
            else None
        ),
        "latency_ms": round(sum(call["latency_ms"] for call in calls), 3),
        "estimated_cost": total("estimated_cost"),
    }


class RunTelemetry:
    """Append-only per-call records plus a small run summary."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(
        self,
        *,
        stage: str,
        provider: str | None,
        model: str | None,
        response: Any = None,
        latency_ms: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        usage = _usage(response) if success else {
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
        }
        self.calls.append({
            "stage": stage,
            "provider": provider,
            "model": model,
            **usage,
            "latency_ms": round(latency_ms, 3),
            "estimated_cost": estimate_cost(
                provider,
                model,
                usage["input_tokens"],
                usage["output_tokens"],
            ) if success else None,
            "success": success,
            "error_type": error_type,
        })

    def envelope(self) -> dict[str, Any]:
        stages = {}
        for call in self.calls:
            stages.setdefault(call["stage"], []).append(call)
        return {
            "calls": self.calls,
            "summary": {
                **_aggregate(self.calls),
                "failure_stage": next(
                    (call["stage"] for call in self.calls if not call["success"]),
                    None,
                ),
                "per_stage": {
                    stage: _aggregate(stage_calls)
                    for stage, stage_calls in stages.items()
                },
            },
        }


class TelemetryClient:
    """Thin client wrapper that preserves the injected client's interface."""

    def __init__(
        self,
        client: Any,
        telemetry: RunTelemetry,
        *,
        stage: str,
        model: str = "",
    ) -> None:
        self.client = client
        self.telemetry = telemetry
        self.stage = stage
        self.model = model

    def create(self, payload: Mapping[str, Any]) -> Any:
        request_payload = dict(payload)
        instructions = request_payload.get("instructions", "")
        request_payload["instructions"] = (
            f"{instructions}\n\n{TRUST_BOUNDARY_INSTRUCTIONS}"
        )
        provider = _provider_name(self.client)
        model = request_payload.get("model") or self.model or getattr(self.client, "model", None)
        started = perf_counter()
        try:
            response = self.client.create(request_payload)
        except Exception as exc:
            self.telemetry.record(
                stage=self.stage,
                provider=provider,
                model=model,
                latency_ms=(perf_counter() - started) * 1000,
                success=False,
                error_type=type(exc).__name__,
            )
            raise
        self.telemetry.record(
            stage=self.stage,
            provider=provider,
            model=model,
            response=response,
            latency_ms=(perf_counter() - started) * 1000,
            success=True,
        )
        return response


__all__ = ["MODEL_PRICING", "RunTelemetry", "TelemetryClient", "estimate_cost"]
