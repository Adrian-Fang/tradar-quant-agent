"""Thin single-turn application entrypoint for the Tradar Agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .agent import run_agent
from .core.providers import (
    DeepSeekChatClient,
    OllamaEmbeddingClient,
    OpenAIEmbeddingClient,
    OpenAIResponsesClient,
)


DEFAULT_MODELS = {
    "deepseek": "deepseek-flash",
    "openai": "gpt-5",
}


def create_provider_client(provider: str, *, api_key: str | None = None) -> Any:
    """Create the chat client used by all model-backed runtime stages."""
    if provider == "deepseek":
        return DeepSeekChatClient(api_key=api_key)
    if provider == "openai":
        return OpenAIResponsesClient(api_key=api_key)
    raise ValueError(f"unsupported provider: {provider}")


def _default_model(provider: str) -> str:
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL", DEFAULT_MODELS[provider])
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", DEFAULT_MODELS[provider])
    return ""


def run_request(
    user_request: str,
    *,
    provider: str = "deepseek",
    model: str = "",
    retrieval: str = "none",
    client: Any | None = None,
    embedding_client: Any | None = None,
    api_key: str | None = None,
    **run_options: Any,
) -> dict[str, Any]:
    """Run one request through the existing integrated Agent runtime."""
    if provider not in {"deepseek", "openai", "fixture"}:
        raise ValueError(f"unsupported provider: {provider}")
    if retrieval not in {"none", "ollama", "openai"}:
        raise ValueError(f"unsupported retrieval mode: {retrieval}")
    if provider == "fixture" and client is None:
        raise ValueError("fixture provider requires an injected client")

    if client is None:
        client = create_provider_client(provider, api_key=api_key)
    model = model or _default_model(provider)

    semantic_embedder = None
    if retrieval == "ollama":
        semantic_embedder = (
            embedding_client
            if embedding_client is not None
            else OllamaEmbeddingClient()
        )
    elif retrieval == "openai":
        semantic_embedder = (
            embedding_client
            if embedding_client is not None
            else OpenAIEmbeddingClient()
        )

    return run_agent(
        user_request,
        planner_client=client,
        hitl_client=client,
        synthesis_client=client,
        grounding_client=client,
        retrieval_client=client if semantic_embedder is not None else None,
        semantic_embedder=semantic_embedder,
        model=model,
        **run_options,
    )


def _json_default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def format_json(result: dict[str, Any]) -> str:
    return json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        default=_json_default,
    )


def format_human(result: dict[str, Any]) -> str:
    observed = result.get("observed") or {}
    outcome = (observed.get("outcome") or {}).get("status", "-")
    summary = (result.get("telemetry") or {}).get("summary") or {}
    cost = summary.get("estimated_cost")
    currency = summary.get("estimated_cost_currency")
    cost_text = "unknown" if cost is None else f"{cost} {currency or ''}".strip()
    answer = result.get("answer") or "-"

    lines = [
        f"status: {result.get('status', '-')}",
        f"outcome: {outcome}",
        f"answer: {answer}",
        (
            "telemetry: "
            f"calls={summary.get('calls', '-')} "
            f"tokens={summary.get('total_tokens', 'unknown')} "
            f"estimated_cost={cost_text} "
            f"wall_clock_ms={summary.get('wall_clock_ms', 'unknown')}"
        ),
        (
            "stages: "
            f"failure={summary.get('failure_stage') or '-'} "
            f"terminal={summary.get('terminal_stage') or '-'}"
        ),
    ]
    if result.get("error"):
        lines.append(f"error: {result['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Tradar Agent research request")
    parser.add_argument("user_request")
    parser.add_argument("--provider", choices=("deepseek", "openai"), default="deepseek")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--retrieval",
        choices=("none", "ollama", "openai"),
        default="none",
        help="optional semantic retrieval embedding provider",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        result = run_request(
            args.user_request,
            provider=args.provider,
            model=args.model,
            retrieval=args.retrieval,
        )
    except Exception as exc:
        if args.as_json:
            print(json.dumps({
                "status": "error",
                "error_type": "configuration_error",
                "error": f"{type(exc).__name__}: {exc}",
            }, ensure_ascii=False))
        else:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(format_json(result) if args.as_json else format_human(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MODELS",
    "create_provider_client",
    "format_human",
    "format_json",
    "main",
    "run_request",
]
