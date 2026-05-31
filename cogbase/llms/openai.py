"""OpenAI-compatible implementation of :class:`LLMBase`. 

This covers OpenAI, Anthropic's compatibility endpoint, vLLM, Ollama,
and any other compatible server.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

from cogbase.llms.base import (
    ChatMessage,
    CompletionResult,
    LLMBase,
    ReasoningEffort,
    ToolCall,
    ToolDefinition,
)


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


# Models that reject the temperature parameter entirely.
_NO_TEMPERATURE_MODELS: frozenset[str] = frozenset({"gpt-5.5"})


class OpenAILLM(LLMBase):
    """LLM backend that calls ``client.chat.completions.create``."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        mini_model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._mini_model = mini_model
        self._reasoning_effort = reasoning_effort

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        model: str | None = None,
    ) -> CompletionResult:
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            model=model,
            stream=False,
        )
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls: list[ToolCall] | None = None
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in choice.message.tool_calls
            ]

        raw_content = choice.message.content
        content = _coerce_message_content(raw_content).strip() if raw_content else None
        return CompletionResult(content=content, tool_calls=tool_calls)

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str | CompletionResult, None]:
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            model=model,
            stream=True,
        )
        stream = await self._client.chat.completions.create(**kwargs)
        tool_call_accum: dict[int, dict] = {}
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
            for tc_delta in delta.tool_calls or []:
                idx = tc_delta.index
                if idx not in tool_call_accum:
                    tool_call_accum[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.id:
                    tool_call_accum[idx]["id"] = tc_delta.id
                if tc_delta.function and tc_delta.function.name:
                    tool_call_accum[idx]["name"] = tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    tool_call_accum[idx]["arguments"] += tc_delta.function.arguments
        if tool_call_accum:
            yield CompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(id=v["id"], name=v["name"], arguments=v["arguments"])
                    for _, v in sorted(tool_call_accum.items())
                ],
            )

    def _resolve_model(self, model: str | None) -> str:
        if model == "mini":
            return self._mini_model or self._model
        return model or self._model

    def _build_kwargs(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None,
        max_tokens: int | None,
        temperature: float | None,
        reasoning_effort: ReasoningEffort | None,
        model: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        resolved_model = self._resolve_model(model)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None and resolved_model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature
        effective_reasoning_effort = reasoning_effort or self._reasoning_effort
        if effective_reasoning_effort is not None:
            kwargs["reasoning_effort"] = effective_reasoning_effort
        # for benchmarks, when using gpt-5.4+ models, manually set service_tier to "flex" (half model price).
        # gpt-4o-mini does not recognize "flex", don't set it.
        # kwargs["service_tier"] = "flex"
        return kwargs
