"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from cogbase.config.prompt import ConfigPromptMixin


class LLMConfig(ConfigPromptMixin, BaseModel):
    provider: Literal["openai", "openai-compatible"] = Field(
        default="openai",
        description=(
            "LLM provider. 'openai' targets the official OpenAI API. "
            "'openai-compatible' targets any OpenAI-compatible server "
            "(vLLM, Alibaba DashScope, etc.) — requires base_url."
        ),
    )
    model: str = Field(description="Model name to use for LLM calls.")
    mini_model: str | None = Field(
        default=None,
        description=(
            "Optional smaller/faster model for lightweight calls. "
            "Callers request it via model='mini'; falls back to 'model' when unset."
        ),
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for the API endpoint. Required when provider is "
            "'openai-compatible'. Examples: "
            "'https://dashscope.aliyuncs.com/compatible-mode/v1' (Alibaba DashScope), "
            "'http://localhost:8000/v1' (vLLM)."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description="Explicit API key. Takes priority over api_key_env and the OPENAI_API_KEY fallback.",
    )
    api_key_env: str | None = Field(
        default=None,
        description=(
            "Name of the environment variable holding the API key. "
            "Checked when api_key is not set. "
            "Example: 'DASHSCOPE_API_KEY' for Alibaba DashScope."
        ),
    )

    @model_validator(mode="after")
    def _check_base_url(self) -> "LLMConfig":
        if self.provider == "openai-compatible" and not self.base_url:
            raise ValueError("base_url is required when provider is 'openai-compatible'")
        return self

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return os.environ.get("OPENAI_API_KEY")


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
