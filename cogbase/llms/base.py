"""Abstract contract for chat-completion LLM backends."""

from __future__ import annotations

import abc
from collections.abc import AsyncGenerator
from typing import Literal, TypedDict

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ChatMessage(TypedDict):
    """A chat message for completion backends."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LLMBase(abc.ABC):
    """Provider-agnostic async chat completion interface."""

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        """Return one full completion for *messages*."""

    @abc.abstractmethod
    async def complete_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion deltas for *messages*."""
