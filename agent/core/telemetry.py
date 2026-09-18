"""Small provider-call telemetry contract for one Agent run."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

# Prices are explicit, versioned configuration, separate from provider usage.
PRICING_VERSION = "2026-09-18"
MODEL_PRICING = {
    ("deepseek", "deepseek-flash"): {
        "currency": "CNY",
        "off_peak": {
            "input_per_million": 1.0,
            "cached_input_per_million": 0.02,
            "output_per_million": 4.0,
        },
        "peak": {
            "input_per_million": 2.0,
            "cached_input_per_million": 0.04,
            "output_per_million": 8.0,
        },
    },
    ("deepseek", "deepseek-v4-pro"): {
        "currency": "CNY",
        "off_peak": {
            "input_per_million": 4.5,
            "cached_input_per_million": 0.15,
            "output_per_million": 13.5,
        },
        "peak": {
            "input_per_million": 9.0,
            "cached_input_per_million": 0.30,
            "output_per_million": 27.0,
        },
    },
    ("openai", "gpt-5"): {
        "currency": "USD",
        "input_per_million": 1.25,
        "cached_input_per_million": 0.125,
        "output_per_million": 10.0,
    },
}
DEEPSEEK_MODEL_ALIASES = {
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
}
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


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
    names = {
        "DeepSeekChatClient": "deepseek",
        "OpenAIResponsesClient": "openai",
        "FixtureClient": "fixture",
    }
    seen = set()
    while client is not None and id(client) not in seen:
        seen.add(id(client))
        provider = getattr(client, "provider", None)
        if isinstance(provider, str) and provider:
            return provider
        provider = names.get(type(client).__name__)
        if provider:
            return provider
        client = getattr(client, "client", None)
    return None


def _canonical_model(provider: str | None, model: str) -> str:
    if provider == "deepseek":
        return DEEPSEEK_MODEL_ALIASES.get(model, model)
    return model


def estimate_cost(
    provider: str | None,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int | None = None,
    at: datetime | None = None,
) -> float | None:
    if provider is None or model is None or input_tokens is None or output_tokens is None:
        return None
    canonical_model = _canonical_model(provider, model)
    pricing = MODEL_PRICING.get((provider, canonical_model))
    if pricing is None:
        return None
    if cached_tokens is None or cached_tokens < 0 or cached_tokens > input_tokens:
        return None
    if "peak" in pricing:
        local_time = at or datetime.now(BEIJING_TIMEZONE)
        local_time = (
            local_time.replace(tzinfo=BEIJING_TIMEZONE)
            if local_time.tzinfo is None
            else local_time.astimezone(BEIJING_TIMEZONE)
        )
        is_peak = (
            local_time.weekday() < 5
            and (9 <= local_time.hour < 12 or 14 <= local_time.hour < 18)
        )
        pricing = pricing["peak" if is_peak else "off_peak"]
    return round(
        (input_tokens - cached_tokens) * pricing["input_per_million"] / 1_000_000
        + cached_tokens * pricing.get(
            "cached_input_per_million", pricing["input_per_million"]
        ) / 1_000_000
        + output_tokens * pricing["output_per_million"] / 1_000_000,
        8,
    )


def _aggregate(calls: list[dict[str, Any]]) -> dict[str, Any]:
    def total(field: str) -> int | float | None:
        values = [call[field] for call in calls]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    estimated_cost = total("estimated_cost")
    currencies = {
        call["estimated_cost_currency"]
        for call in calls
        if call["estimated_cost"] is not None
    }
    if len(currencies) > 1:
        estimated_cost = None
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
        "provider_latency_ms": round(sum(call["latency_ms"] for call in calls), 3),
        "estimated_cost": estimated_cost,
        "estimated_cost_currency": (
            next(iter(currencies))
            if estimated_cost is not None and len(currencies) == 1
            else None
        ),
    }


class RunTelemetry:
    """Append-only per-call records plus a small run summary."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.started = perf_counter()
        self.runtime_failure_stage: str | None = None
        self.terminal_stage: str | None = None

    def set_runtime_stages(
        self,
        failure_stage: str | None,
        terminal_stage: str | None,
    ) -> None:
        self.runtime_failure_stage = failure_stage
        self.terminal_stage = terminal_stage

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
        at: datetime | None = None,
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
                usage["cached_tokens"],
                at,
            ) if success else None,
            "estimated_cost_currency": (
                MODEL_PRICING.get(
                    (provider, _canonical_model(provider, model or "")), {}
                ).get("currency")
                if success else None
            ),
            "success": success,
            "error_type": error_type,
        })

    def envelope(self) -> dict[str, Any]:
        stages = {}
        for call in self.calls:
            stages.setdefault(call["stage"], []).append(call)
        provider_failure_stage = next(
            (call["stage"] for call in self.calls if not call["success"]),
            None,
        )
        return {
            "calls": self.calls,
            "summary": {
                **_aggregate(self.calls),
                "pricing_version": PRICING_VERSION,
                "wall_clock_ms": round((perf_counter() - self.started) * 1000, 3),
                "failure_stage": self.runtime_failure_stage or provider_failure_stage,
                "provider_failure_stage": provider_failure_stage,
                "terminal_stage": self.terminal_stage,
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
        provider = _provider_name(self.client)
        model = payload.get("model") or self.model or getattr(self.client, "model", None)
        started = perf_counter()
        request_time = datetime.now(BEIJING_TIMEZONE)
        try:
            response = self.client.create(payload)
        except Exception as exc:
            self.telemetry.record(
                stage=self.stage,
                provider=provider,
                model=model,
                latency_ms=(perf_counter() - started) * 1000,
                success=False,
                error_type=type(exc).__name__,
                at=request_time,
            )
            raise
        self.telemetry.record(
            stage=self.stage,
            provider=provider,
            model=model,
            response=response,
            latency_ms=(perf_counter() - started) * 1000,
            success=True,
            at=request_time,
        )
        return response


__all__ = [
    "BEIJING_TIMEZONE",
    "DEEPSEEK_MODEL_ALIASES",
    "MODEL_PRICING",
    "PRICING_VERSION",
    "RunTelemetry",
    "TelemetryClient",
    "estimate_cost",
]
