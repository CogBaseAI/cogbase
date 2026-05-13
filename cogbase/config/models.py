"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from cogbase.config.prompt import ConfigPromptMixin


class LLMConfig(ConfigPromptMixin, BaseModel):
    provider: Literal["openai"] = Field(
        default="openai",
        description="LLM provider to use."
    )
    model: str = Field(description="Model name to use for LLM calls.")
    mini_model: str | None = Field(
        default=None,
        description=(
            "Optional smaller/faster model for lightweight calls. "
            "Callers request it via model='mini'; falls back to 'model' when unset."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key. Falls back to OPENAI_API_KEY when omitted.",
    )

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get("OPENAI_API_KEY")


class EmbeddingConfig(ConfigPromptMixin, BaseModel):
    provider: Literal["openai", "sentence-transformers"] = Field(
        default="openai",
        description="Embedding provider to use."
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name."
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key. Falls back to OPENAI_API_KEY when omitted.",
    )
    dimensions: int | None = Field(
        default=None,
        description="Optional output vector dimension override.",
    )
