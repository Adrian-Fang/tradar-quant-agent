"""Shared remote provider adapters used by capability eval runners."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from typing import Any
from urllib import request

from openai import OpenAI


class OllamaEmbeddingClient:
    """Local Ollama embeddings adapter using the /api/embed endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )).rstrip("/")
        self.model = model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"
        )
        self.timeout = timeout

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
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
        timeout: float = 60.0,
        sdk_client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for semantic retrieval")
        self.model = model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        client_args: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": timeout,
        }
        if base_url := os.getenv("OPENAI_BASE_URL"):
            client_args["base_url"] = base_url
        self.client = sdk_client or OpenAI(**client_args)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]


class OpenAIResponsesClient:
    """OpenAI Responses client using the installed OpenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 60.0,
        sdk_client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI tool calling")
        self.endpoint = endpoint or os.getenv(
            "OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses"
        )
        base_url = self.endpoint.removesuffix("/responses")
        self.client = sdk_client or OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def create(self, payload: Mapping[str, Any]) -> Any:
        return self.client.responses.create(**dict(payload))


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
        timeout: float = 60.0,
        sdk_client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek tool calling")
        if endpoint and not base_url:
            base_url = endpoint.removesuffix("/chat/completions")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.client = sdk_client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

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
        response = self.client.chat.completions.create(**request)
        choices = getattr(response, "choices", [])
        if not choices:
            raise RuntimeError("DeepSeek response did not contain a choice")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("DeepSeek response did not contain a message")
        output = []
        for call in getattr(message, "tool_calls", []) or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            output.append({
                "type": "function_call",
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", "{}"),
                "call_id": getattr(call, "id", None),
            })
        return {
            "output": output,
            "output_text": getattr(message, "content", "") or "",
        }


__all__ = [
    "DeepSeekChatClient",
    "OllamaEmbeddingClient",
    "OpenAIEmbeddingClient",
    "OpenAIResponsesClient",
]
