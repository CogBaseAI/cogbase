"""Tests for the LLM base contract."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from cogbase.llms import LLMBase
from cogbase.llms.base import ChatMessage, ReasoningEffort


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
                max_tokens: int | None = None,
                temperature: float | None = None,
                reasoning_effort: ReasoningEffort | None = None,
            ) -> str:
                _ = (messages, max_tokens, temperature, reasoning_effort)
                return "ok"

            async def complete_stream(
                self,
                messages: list[ChatMessage],
                *,
                max_tokens: int | None = None,
                temperature: float | None = None,
                reasoning_effort: ReasoningEffort | None = None,
            ) -> AsyncGenerator[str, None]:
                _ = (messages, max_tokens, temperature, reasoning_effort)
                yield "o"
                yield "k"

        llm = ConstantLLM()
        messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]
        assert await llm.complete(messages) == "ok"
        assert [part async for part in llm.complete_stream(messages)] == ["o", "k"]
