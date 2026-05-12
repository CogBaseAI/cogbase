"""Abstract contract for chat-completion LLM backends."""

from __future__ import annotations

import abc
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Literal, TypedDict, Union

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ToolCall(TypedDict):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: str  # JSON-encoded


class ToolDefinition(TypedDict):
    """A tool the LLM may invoke."""

    name: str
    description: str
    parameters: dict  # JSON Schema object


ToolHandler = Callable[[dict], Union[Awaitable[str], str]]


class SystemTool:
    """A named tool with its JSON-schema definition and async/sync handler.

    Args:
        definition: ``ToolDefinition`` exposed to the LLM.
        handler:    ``(inputs: dict) -> str | Awaitable[str]`` — executes the
                    tool and returns a result string for the LLM.
    """

    def __init__(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self.definition = definition
        self.handler = handler

    @property
    def name(self) -> str:
        return self.definition["name"]


class CompletionResult(TypedDict):
    """Return value of :meth:`LLMBase.complete`."""

    content: str | None
    tool_calls: list[ToolCall] | None


class _ChatMessageRequired(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]


class ChatMessage(_ChatMessageRequired, total=False):
    """A chat message for completion backends.

    Optional fields:
    - ``content``: text content (absent when the assistant returns only tool calls)
    - ``tool_call_id``: set on ``role="tool"`` result messages
    - ``tool_calls``: set on ``role="assistant"`` messages that requested tool calls
    """

    content: str
    tool_call_id: str
    tool_calls: list[ToolCall]


class LLMBase(abc.ABC):
    """Provider-agnostic async chat completion interface."""

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> CompletionResult:
        """Return one full completion for *messages*.

        When *tools* are provided the model may return tool calls instead of
        (or in addition to) text content.  Callers should check
        ``result["tool_calls"]`` before ``result["content"]``.
        """

    @abc.abstractmethod
    async def complete_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> AsyncGenerator[str | CompletionResult, None]:
        """Stream a completion for *messages*.

        Yields ``str`` deltas for text content.  If the model requests tool
        calls instead of (or after) text, a single ``CompletionResult`` is
        yielded at the end with ``tool_calls`` populated and ``content=None``.
        Callers should check ``isinstance(chunk, dict)`` (TypedDict) to
        distinguish the final result from text deltas.
        """
