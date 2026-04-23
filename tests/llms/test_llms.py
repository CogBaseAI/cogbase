"""Tests for the LLM base contract."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from cogbase.llms import LLMBase
from cogbase.llms.base import ChatMessage, CompletionResult, ReasoningEffort, ToolDefinition


class TestLLMBaseIsAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            LLMBase()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_custom_backend_satisfies_contract(self) -> None:
        class ConstantLLM(LLMBase):
            async def complete(
                self,
                messages: list[ChatMessage],
                *,
                tools: list[ToolDefinition] | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
                reasoning_effort: ReasoningEffort | None = None,
            ) -> CompletionResult:
                _ = (messages, tools, max_tokens, temperature, reasoning_effort)
                return CompletionResult(content="ok", tool_calls=None)

            async def complete_stream(
                self,
                messages: list[ChatMessage],
                *,
                tools: list[ToolDefinition] | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
                reasoning_effort: ReasoningEffort | None = None,
            ) -> AsyncGenerator[str, None]:
                _ = (messages, tools, max_tokens, temperature, reasoning_effort)
                yield "o"
                yield "k"

        llm = ConstantLLM()
        messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]
        result = await llm.complete(messages)
        assert result["content"] == "ok"
        assert result["tool_calls"] is None
        assert [part async for part in llm.complete_stream(messages)] == ["o", "k"]
