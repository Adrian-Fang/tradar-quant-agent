"""Shared remote provider adapters used by capability eval runners."""

from __future__ import annotations

from collections.abc import Mapping
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import inspect
import json
import math
import os
import random
from time import perf_counter
from typing import Any
from urllib import request

import httpx
from openai import AsyncOpenAI, OpenAI as SyncOpenAI, APIConnectionError, APIResponseValidationError, APITimeoutError, RateLimitError

_ASYNC_SDK_TYPE = AsyncOpenAI


ATTEMPT_TIMEOUT_SECONDS = 20.0
CALL_TIMEOUT_SECONDS = 45.0
MAX_RETRIES = 1
MAX_RETRY_DELAY_SECONDS = 1.0


class ProviderError(RuntimeError):
    """Operational failure, separate from a model's semantic abstention."""

    def __init__(self, code: str, message: str, *, attempts=None):
        super().__init__(message)
        self.code = code
        self.attempts = attempts or []


def provider_error_type(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        return exc.code
    if isinstance(exc, (TimeoutError, APITimeoutError, httpx.TimeoutException)) or getattr(exc, "status_code", None) == 408:
        return "provider_timeout"
    if isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429:
        return "provider_rate_limit"
    if isinstance(exc, json.JSONDecodeError) and not exc.doc.strip():
        return "provider_empty_response"
    if isinstance(exc, (APIResponseValidationError, json.JSONDecodeError)):
        return "provider_invalid_response"
    return "provider_error"


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _retry_delay(exc: Exception) -> float | None:
    """Honor short server delays; decline long waits instead of retrying early."""
    headers = getattr(getattr(exc, "response", None), "headers", {})
    raw = headers.get("retry-after-ms") or headers.get("retry-after")
    if raw is not None:
        try:
            delay = float(raw) / (1000 if headers.get("retry-after-ms") else 1)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError):
                delay = -1
        if math.isfinite(delay) and delay >= 0:
            return delay if delay <= MAX_RETRY_DELAY_SECONDS else None
    return random.uniform(0.25, 0.5)


def _sdk_call(sdk_client, client_args, resource, payload, normalize, timeout):
    """One SDK operation with explicit retries, cancellation and no hidden retries."""
    if isinstance(sdk_client, SyncOpenAI):
        raise ProviderError("provider_error", "bounded network calls require an async SDK client")
    attempts = []

    async def request():
        owned = sdk_client is None
        client = sdk_client or AsyncOpenAI(**client_args, max_retries=0, timeout=timeout)
        if not owned and isinstance(client, _ASYNC_SDK_TYPE):
            client = client.with_options(max_retries=0, timeout=timeout)
        try:
            deadline = perf_counter() + CALL_TIMEOUT_SECONDS
            for index in range(MAX_RETRIES + 1):
                response = None
                started = perf_counter()
                try:
                    async with asyncio.timeout(min(timeout, max(0, deadline - started))):
                        endpoint = client
                        for name in resource:
                            endpoint = getattr(endpoint, name)
                        value = endpoint.create(**payload)
                        response = await value if inspect.isawaitable(value) else value
                        normalized = normalize(response)
                except Exception as exc:
                    code = provider_error_type(exc)
                    attempts.append({"attempt": index + 1, "latency_ms": (perf_counter() - started) * 1000,
                                     "success": False, "error_type": code, "usage": _field(response, "usage")})
                    retryable = (not isinstance(exc, ProviderError) and (
                                 isinstance(exc, (TimeoutError, APITimeoutError, httpx.TimeoutException, APIConnectionError))
                                 or getattr(exc, "status_code", None) in {408, 409, 429}
                                 or (getattr(exc, "status_code", 0) or 0) >= 500))
                    # Quota/billing errors require intervention, not another request.
                    if getattr(exc, "code", None) in {"insufficient_quota", "billing_hard_limit_reached"}:
                        retryable = False
                    delay = _retry_delay(exc) if retryable else None
                    if index < MAX_RETRIES and delay is not None and delay < deadline - perf_counter():
                        attempts[-1]["backoff_ms"] = delay * 1000
                        await asyncio.sleep(delay)
                        continue
                    raise ProviderError(code, f"{code}: provider request failed", attempts=attempts) from exc
                attempts.append({"attempt": index + 1, "latency_ms": (perf_counter() - started) * 1000,
                                 "success": True, "error_type": None, "usage": _field(response, "usage")})
                normalized["provider_call"] = {"attempts": attempts}
                return normalized
        finally:
            if owned:
                # Cleanup must not replace the classified request failure.
                try:
                    await asyncio.wait_for(client.close(), timeout=1)
                except Exception:
                    pass

    # HTTP runtime calls this synchronous adapter in its existing worker thread.
    # Don't let asyncio.run's executor shutdown wait minutes for a canceled OS DNS lookup.
    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(executor)
    try:
        return loop.run_until_complete(request())
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        loop.close()


class OllamaEmbeddingClient:
    """Local Ollama embeddings adapter using the /api/embed endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        truncate: bool | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )).rstrip("/")
        self.model = model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"
        )
        self.timeout = timeout
        self.truncate = truncate

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        if self.truncate is not None:
            payload["truncate"] = self.truncate
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama response contained an invalid embeddings batch")
        return embeddings


class OpenAIEmbeddingClient:
    """OpenAI embeddings adapter shared by retrieval capabilities."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = ATTEMPT_TIMEOUT_SECONDS,
        sdk_client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for semantic retrieval")
        self.model = model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.client_args: dict[str, Any] = {
            "api_key": self.api_key,
        }
        if base_url := os.getenv("OPENAI_BASE_URL"):
            self.client_args["base_url"] = base_url
        self.timeout = _bounded_timeout(timeout)
        self.client = sdk_client

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        def normalize(response):
            data = _field(response, "data")
            if not isinstance(data, list) or len(data) != len(texts):
                raise ProviderError("provider_invalid_response", "provider embeddings batch is invalid")
            indices = [_field(item, "index") for item in data]
            if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(texts))):
                raise ProviderError("provider_invalid_response", "provider embedding indices are invalid")
            vectors = [_field(item, "embedding") for item in sorted(data, key=lambda item: _field(item, "index"))]
            if any(not isinstance(vector, list) or not vector for vector in vectors):
                raise ProviderError("provider_invalid_response", "provider embeddings are invalid")
            return {"embeddings": vectors}

        return _sdk_call(self.client, self.client_args, ("embeddings",),
                         {"model": self.model, "input": texts}, normalize, self.timeout)["embeddings"]


class OpenAIResponsesClient:
    """OpenAI Responses client using the installed OpenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = ATTEMPT_TIMEOUT_SECONDS,
        sdk_client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI tool calling")
        self.endpoint = endpoint or os.getenv(
            "OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses"
        )
        base_url = self.endpoint.removesuffix("/responses")
        self.client_args = {"api_key": self.api_key, "base_url": base_url}
        self.timeout = _bounded_timeout(timeout)
        self.client = sdk_client

    def create(self, payload: Mapping[str, Any]) -> Any:
        return _sdk_call(self.client, self.client_args, ("responses",),
                         dict(payload), _normalize_responses, self.timeout)


def _deepseek_tools(schemas: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses-style schemas to Chat Completions format."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
        for schema in schemas
    ]


class DeepSeekChatClient:
    """DeepSeek OpenAI-compatible Chat Completions client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        endpoint: str | None = None,
        timeout: float = ATTEMPT_TIMEOUT_SECONDS,
        sdk_client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek tool calling")
        if endpoint and not base_url:
            base_url = endpoint.removesuffix("/chat/completions")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.client_args = {"api_key": self.api_key, "base_url": self.base_url}
        self.timeout = _bounded_timeout(timeout)
        self.client = sdk_client

    def create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            "model": payload["model"],
            "messages": [
                {"role": "system", "content": payload["instructions"]},
                {"role": "user", "content": payload["input"]},
            ],
            "temperature": 0.0,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if payload.get("tools"):
            request.update(tools=_deepseek_tools(payload["tools"]), tool_choice="required")
        return _sdk_call(self.client, self.client_args, ("chat", "completions"),
                         request, _normalize_deepseek, self.timeout)


def _bounded_timeout(timeout):
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("provider timeout must be a positive finite number")
    return min(float(timeout), ATTEMPT_TIMEOUT_SECONDS)


def _normalize_responses(response):
    if response is None or isinstance(response, str) and not response.strip():
        raise ProviderError("provider_empty_response", "provider returned no output")
    if isinstance(response, Mapping):
        normalized = dict(response)
    elif callable(getattr(response, "model_dump", None)):
        normalized = response.model_dump()
        normalized["output_text"] = _field(response, "output_text", "")
    else:
        raise ProviderError("provider_invalid_response", "provider response envelope is invalid")
    output, text = normalized.get("output", []), normalized.get("output_text", "")
    if not isinstance(output, list) or not isinstance(text, str):
        raise ProviderError("provider_invalid_response", "provider output fields are invalid")
    if not output and not text.strip():
        raise ProviderError("provider_empty_response", "provider returned no output")
    return normalized


def _normalize_deepseek(response):
    if response is None or isinstance(response, str) and not response.strip():
        raise ProviderError("provider_empty_response", "provider returned no output")
    choices = _field(response, "choices")
    if not isinstance(choices, list):
        raise ProviderError("provider_invalid_response", "provider choices field is invalid")
    if not choices:
        raise ProviderError("provider_empty_response", "provider returned no choice")
    message = _field(choices[0], "message")
    if message is None:
        raise ProviderError("provider_invalid_response", "provider choice contains no message")
    output = []
    calls = _field(message, "tool_calls")
    text = _field(message, "content")
    calls = [] if calls is None else calls
    text = "" if text is None else text
    if not isinstance(calls, list) or not isinstance(text, str):
        raise ProviderError("provider_invalid_response", "provider message fields are invalid")
    for call in calls:
        function = _field(call, "function")
        name, arguments = _field(function, "name"), _field(function, "arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, str) or not arguments.strip():
            raise ProviderError("provider_invalid_response", "provider tool call is invalid")
        output.append({
            "type": "function_call", "name": name, "arguments": arguments,
            "call_id": _field(call, "id"),
        })
    if not output and not text.strip():
        raise ProviderError("provider_empty_response", "provider returned no text or tool calls")
    normalized = {"output": output, "output_text": text}
    usage = _field(response, "usage")
    if usage is not None:
        normalized["usage"] = usage
    return normalized


__all__ = [
    "DeepSeekChatClient",
    "OllamaEmbeddingClient",
    "OpenAIEmbeddingClient",
    "OpenAIResponsesClient",
    "ProviderError",
    "provider_error_type",
]
