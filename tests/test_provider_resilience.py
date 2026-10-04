"""Bounded remote adapters: no network, real SDK transport where relevant."""

import asyncio
import json
import threading
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI, OpenAI, APITimeoutError, RateLimitError
import pytest

from agent.core import providers
from agent.core.providers import DeepSeekChatClient, OpenAIEmbeddingClient, OpenAIResponsesClient, ProviderError
from agent.core.telemetry import RunTelemetry, TelemetryClient


PAYLOAD = {"model": "deepseek-flash", "instructions": "return JSON", "input": "query"}
GOOD = {"choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_cache_hit_tokens": 0}}


class SDK:
    def __init__(self, *responses):
        self.responses_queue = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.responses = self.embeddings = SimpleNamespace(create=self.create)
        self.close = AsyncMock()

    async def create(self, **payload):
        self.calls.append(payload)
        value = self.responses_queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture(autouse=True)
def no_random_wait(monkeypatch):
    monkeypatch.setattr(providers.random, "uniform", lambda *_: 0)
    # SDK's one-time OS probe uses thread wakeups blocked in this test host.
    # Only that probe is stubbed; requests still use the actual SDK/HTTP transport.
    monkeypatch.setattr("openai._base_client.asyncify", lambda _: AsyncMock(return_value="linux"))


def adapter(sdk, **options):
    return DeepSeekChatClient(api_key="test-key", sdk_client=sdk, **options)


def test_deepseek_auto_choice_preserves_native_tool_conversation():
    sdk = SDK(GOOD)
    conversation = [
        {"role": "user", "content": "Inspect."},
        {"type": "function_call", "call_id": "call-1", "name": "inspect_universe", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-1", "output": '{"count":12}'},
    ]
    adapter(sdk).create({**PAYLOAD, "input": conversation, "tool_choice": "auto", "tools": [{
        "type": "function", "name": "inspect_universe", "description": "Inspect.", "parameters": {"type": "object"},
    }]})
    request = sdk.calls[0]
    assert request["tool_choice"] == "auto" and request["parallel_tool_calls"] is False
    assert request["messages"][1] == {"role": "user", "content": "Inspect."}
    assert request["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert request["messages"][3] == {"role": "tool", "tool_call_id": "call-1", "content": '{"count":12}'}


def test_openai_keeps_native_input_and_auto_choice():
    sdk = SDK({"output_text": "42"})
    conversation = [{"role": "user", "content": "17+25?"}]
    result = OpenAIResponsesClient(api_key="test-key", sdk_client=sdk).create({
        **PAYLOAD, "input": conversation, "tool_choice": "auto", "tools": [],
    })
    assert sdk.calls[0]["input"] == conversation and sdk.calls[0]["tool_choice"] == "auto"
    assert result["output_text"] == "42"


def timeout():
    return APITimeoutError(request=httpx.Request("POST", "https://fixture.invalid"))


def rate_limit(delay=None, code="rate_limit"):
    response = httpx.Response(429, request=httpx.Request("POST", "https://fixture.invalid"),
                              headers={} if delay is None else {"retry-after": str(delay)})
    return RateLimitError("fixture", response=response, body={"code": code})


@pytest.mark.parametrize("failure,code", [
    (timeout(), "provider_timeout"), (TimeoutError(), "provider_timeout"),
    (rate_limit(), "provider_rate_limit"),
    (json.JSONDecodeError("bad", "!", 0), "provider_invalid_response"),
    (RuntimeError("other"), "provider_error"),
])
def test_error_classification(failure, code):
    assert providers.provider_error_type(failure) == code


@pytest.mark.parametrize("response,code", [
    ({"choices": []}, "provider_empty_response"),
    ({"choices": [{"message": {"content": " \n"}}]}, "provider_empty_response"),
    ({}, "provider_invalid_response"),
    ({"choices": "wrong"}, "provider_invalid_response"),
    ({"choices": [{}]}, "provider_invalid_response"),
    ({"choices": [{"message": {"content": False}}]}, "provider_invalid_response"),
    ({"choices": [{"message": {"tool_calls": [{"function": {"name": "x", "arguments": None}}]}}]},
     "provider_invalid_response"),
])
def test_invalid_or_empty_provider_envelope_never_retries(response, code):
    sdk = SDK(response, GOOD)
    with pytest.raises(ProviderError) as caught:
        adapter(sdk).create(PAYLOAD)
    assert caught.value.code == code
    assert len(sdk.calls) == len(caught.value.attempts) == 1


@pytest.mark.parametrize("failure", [timeout(), rate_limit()])
def test_transient_failure_retries_once_and_exhaustion_is_controlled(failure):
    sdk = SDK(failure, failure, GOOD)
    with pytest.raises(ProviderError) as caught:
        adapter(sdk).create(PAYLOAD)
    assert len(sdk.calls) == len(caught.value.attempts) == 2
    assert caught.value.code == providers.provider_error_type(failure)
    assert str(caught.value)


@pytest.mark.parametrize("failure", [rate_limit(120), rate_limit(code="insufficient_quota")])
def test_long_retry_after_and_quota_failure_do_not_retry(failure):
    sdk = SDK(failure, GOOD)
    with pytest.raises(ProviderError) as caught:
        adapter(sdk).create(PAYLOAD)
    assert caught.value.code == "provider_rate_limit" and len(sdk.calls) == 1


def test_short_retry_after_is_honored_and_recovery_keeps_attempt_history():
    sdk = SDK(rate_limit(0.001), GOOD)
    result = adapter(sdk).create(PAYLOAD)
    attempts = result["provider_call"]["attempts"]
    assert len(sdk.calls) == len(attempts) == 2
    assert attempts[0]["backoff_ms"] == 1
    assert attempts[0]["error_type"] == "provider_rate_limit"
    assert attempts[1]["success"] and result["output_text"] == "{}"


def test_sdk_factory_has_no_hidden_retries_and_closes_owned_client():
    sdk = SDK(GOOD)
    with patch.object(providers, "AsyncOpenAI", return_value=sdk) as factory:
        client = DeepSeekChatClient(api_key="test-key", timeout=60)
        factory.assert_not_called()  # No network resources before create.
        client.create(PAYLOAD)
    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["timeout"] == client.timeout == 20
    sdk.close.assert_awaited_once()


def test_injected_sync_network_sdk_cannot_bypass_deadline_or_retry_policy():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=GOOD)
    with OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as native:
        with pytest.raises(ProviderError, match="async SDK client"):
            adapter(native).create(PAYLOAD)
    assert requests == []


def test_cleanup_failure_does_not_mask_operational_error():
    sdk = SDK(timeout(), timeout())
    sdk.close.side_effect = RuntimeError("cleanup")
    with patch.object(providers, "AsyncOpenAI", return_value=sdk), pytest.raises(ProviderError) as caught:
        DeepSeekChatClient(api_key="test-key").create(PAYLOAD)
    assert caught.value.code == "provider_timeout" and len(caught.value.attempts) == 2


@pytest.mark.parametrize("status,code,count", [
    (408, "provider_timeout", 2), (429, "provider_rate_limit", 2),
    (503, "provider_error", 2), (401, "provider_error", 1),
])
def test_real_sdk_transport_cannot_add_hidden_retries(status, code, count):
    requests = []
    async def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "fixture", "type": "fixture"}})
    native = AsyncOpenAI(api_key="test-key", max_retries=7,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    try:
        with pytest.raises(ProviderError) as caught:
            adapter(native).create(PAYLOAD)
        assert caught.value.code == code and len(requests) == count
    finally:
        asyncio.run(native.close())


@pytest.mark.parametrize("content,code", [(b"", "provider_empty_response"),
    (b"not JSON", "provider_invalid_response"), (b"{}", "provider_invalid_response")])
def test_real_sdk_invalid_or_empty_response_is_not_semantic_abstention(content, code):
    requests = []
    async def handle(request):
        requests.append(request)
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})
    native = AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    try:
        with pytest.raises(ProviderError) as caught:
            adapter(native).create(PAYLOAD)
        assert caught.value.code == code and len(requests) == 1
    finally:
        asyncio.run(native.close())


def test_real_sdk_success_preserves_normalized_tool_calls_and_usage():
    sent = []
    async def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "type": "function",
                "function": {"name": "respond", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        })
    native = AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    try:
        result = adapter(native).create({**PAYLOAD, "tools": [{"name": "respond", "description": "fixture",
                                                            "parameters": {"type": "object"}}]})
        assert result["output"] == [{"type": "function_call", "name": "respond", "arguments": "{}", "call_id": "call-1"}]
        assert result["usage"].prompt_tokens == 10
        assert sent[0]["tool_choice"] == "required"
        assert len(result["provider_call"]["attempts"]) == 1
    finally:
        asyncio.run(native.close())


def test_attempt_deadline_cancels_inflight_request_and_bounds_total_budget(monkeypatch):
    cancelled = []
    async def hang(**_):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    sdk = SDK()
    sdk.chat.completions.create = hang
    started = perf_counter()
    with pytest.raises(ProviderError) as caught:
        adapter(sdk, timeout=0.01).create(PAYLOAD)
    assert caught.value.code == "provider_timeout" and len(cancelled) == 2
    assert perf_counter() - started < 1
    monkeypatch.setattr(providers, "CALL_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(ProviderError) as caught:
        adapter(sdk, timeout=0.02).create(PAYLOAD)
    assert len(caught.value.attempts) == 1


def test_trickling_http_body_cannot_extend_attempt_deadline():
    streams = []
    class Trickle(httpx.AsyncByteStream):
        closed = False
        async def __aiter__(self):
            while True:
                yield b" "
                await asyncio.sleep(0.001)
        async def aclose(self):
            self.closed = True
    async def handle(request):
        stream = Trickle()
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=stream)
    native = AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    try:
        started = perf_counter()
        with pytest.raises(ProviderError) as caught:
            adapter(native, timeout=0.02).create(PAYLOAD)
        assert caught.value.code == "provider_timeout"
        assert len(streams) == 2 and all(stream.closed for stream in streams)
        assert perf_counter() - started < 1
    finally:
        asyncio.run(native.close())


def test_cancellation_does_not_wait_for_os_dns_executor_shutdown(monkeypatch):
    release = threading.Event()
    async def pending_dns(**_):
        await asyncio.get_running_loop().run_in_executor(None, release.wait)
    sdk = SDK()
    sdk.chat.completions.create = pending_dns
    monkeypatch.setattr(providers, "MAX_RETRIES", 0)
    try:
        started = perf_counter()
        with pytest.raises(ProviderError) as caught:
            adapter(sdk, timeout=0.01).create(PAYLOAD)
        assert caught.value.code == "provider_timeout" and not release.is_set()
        assert perf_counter() - started < 1
    finally:
        release.set()


@pytest.mark.parametrize("retry", [False, True])
def test_telemetry_attempts_latency_and_unknown_usage_after_retry(retry):
    sdk = SDK(*([timeout()] if retry else []), GOOD)
    telemetry = RunTelemetry()
    client = TelemetryClient(adapter(sdk), telemetry, stage="retrieval_verifier")
    client.create(PAYLOAD)
    envelope = telemetry.envelope()
    call, summary = envelope["calls"][0], envelope["summary"]
    assert summary["calls"] == 1 and summary["attempt_count"] == 1 + retry
    assert summary["retry_count"] == retry
    assert summary["per_stage"]["retrieval_verifier"]["provider_latency_ms"] == call["latency_ms"]
    assert call["latency_ms"] >= sum(attempt["latency_ms"] for attempt in call["attempts"]) - 0.01
    assert call["attempts"][-1]["input_tokens"] == 10
    assert call["input_tokens"] == (None if retry else 10)
    assert summary["provider_failure_stage"] is None
    if retry:
        assert call["estimated_cost"] is None
    assert "query" not in json.dumps(envelope) and "test-key" not in json.dumps(envelope)


def test_exhausted_timeout_telemetry_is_error_not_abstain():
    telemetry = RunTelemetry()
    with pytest.raises(ProviderError):
        TelemetryClient(adapter(SDK(timeout(), timeout())), telemetry, stage="planning").create(PAYLOAD)
    envelope = telemetry.envelope()
    assert envelope["calls"][0]["error_type"] == "provider_timeout"
    assert envelope["summary"]["retry_count"] == 1
    assert envelope["summary"]["failure_stage"] == "planning"
    assert envelope["summary"]["total_tokens"] is None


@pytest.mark.parametrize("response,code", [({}, "provider_empty_response"),
    ({"output": "wrong"}, "provider_invalid_response")])
def test_responses_adapter_validates_provider_envelope(response, code):
    with pytest.raises(ProviderError) as caught:
        OpenAIResponsesClient(api_key="test-key", sdk_client=SDK(response)).create(PAYLOAD)
    assert caught.value.code == code


def test_embeddings_share_bounded_policy_and_validate_indices():
    sdk = SDK(timeout(), {"data": [{"index": 0, "embedding": [1.0]}]})
    assert OpenAIEmbeddingClient(api_key="test-key", sdk_client=sdk)(["query"]) == [[1.0]]
    assert len(sdk.calls) == 2
    with pytest.raises(ProviderError) as caught:
        OpenAIEmbeddingClient(api_key="test-key", sdk_client=SDK({"data": [{"embedding": [1.0]}]}))(["query"])
    assert caught.value.code == "provider_invalid_response"
