"""OpenAI-compatible implementation of :class:`LLMBase`. 

This covers OpenAI, Anthropic's compatibility endpoint, vLLM, Ollama,
and any other compatible server.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

from cogbase.llms.base import ChatMessage, LLMBase, ReasoningEffort


def _coerce_message_content(content: Any) -> str:
    """Normalize OpenAI-compatible message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    return str(content)


class OpenAILLM(LLMBase):
    """LLM backend that calls ``client.chat.completions.create``."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        kwargs = self._build_kwargs(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            stream=False,
        )
        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return _coerce_message_content(content).strip()

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> AsyncGenerator[str, None]:
        kwargs = self._build_kwargs(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _build_kwargs(
        self,
        *,
        messages: list[ChatMessage],
        max_tokens: int | None,
        temperature: float | None,
        reasoning_effort: ReasoningEffort | None,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        effective_reasoning_effort = reasoning_effort or self._reasoning_effort
        if effective_reasoning_effort is not None:
            kwargs["reasoning_effort"] = effective_reasoning_effort
        return kwargs
