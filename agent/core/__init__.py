"""Shared agent contracts, providers, and resource loading."""

from dotenv import load_dotenv

from .resources import REPO_ROOT

load_dotenv(REPO_ROOT / ".env", override=False)

from .contracts import ResearchRun, ToolResult, ToolStatus
from .providers import DeepSeekChatClient, OpenAIResponsesClient

__all__ = [
    "DeepSeekChatClient",
    "OpenAIResponsesClient",
    "ResearchRun",
    "ToolResult",
    "ToolStatus",
]
