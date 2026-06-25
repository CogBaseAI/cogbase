"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
    context_window: int = Field(
        default=128_000,
        description=(
            "Context window of 'model', in tokens. Used to size compaction "
            "budgets as a fraction of the window so they can never exceed it."
        ),
    )
    mini_context_window: int | None = Field(
        default=None,
        description=(
            "Context window of 'mini_model', in tokens. Falls back to "
            "'context_window' when unset."
        ),
    )
    base_url: str = Field(
        default='https://api.openai.com/v1',
        description=(
            "Base URL for the API endpoint. Required when provider is "
            "'openai-compatible'. Examples: "
            "'https://dashscope.aliyuncs.com/compatible-mode/v1' (Alibaba DashScope), "
            "'http://localhost:8000/v1' (vLLM)."
        ),
    )
    api_key: str = Field(
        description="API key. Use 'EMPTY' for local servers that require no authentication.",
    )


class EmbeddingConfig(ConfigPromptMixin, BaseModel):
    provider: Literal["openai", "openai-compatible", "sentence-transformers"] = Field(
        default="openai",
        description=(
            "Embedding provider. 'openai' targets the official OpenAI API. "
            "'openai-compatible' targets any OpenAI-compatible embedding server "
            "(vLLM, Alibaba DashScope, etc.) — requires base_url. "
            "'sentence-transformers' runs locally via HuggingFace."
        ),
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name."
    )
    base_url: str = Field(
        default='https://api.openai.com/v1',
        description=(
            "Base URL for the API endpoint. Required when provider is "
            "'openai-compatible'. Examples: "
            "'https://dashscope.aliyuncs.com/compatible-mode/v1' (Alibaba DashScope), "
            "'http://localhost:8000/v1' (vLLM)."
        ),
    )
    api_key: str = Field(
        description="API key. Use 'EMPTY' for local servers that require no authentication.",
    )
    dimensions: int = Field(
        default=1536,
        description="Optional output vector dimension override.",
    )
    batch_size: int = Field(
        default=500,
        ge=1,
        description=(
            "Maximum number of texts sent per embedding API request. A long "
            "document can chunk into thousands of passages, which would exceed "
            "the endpoint's per-request array/token limit; the embedder splits "
            "inputs into sub-batches of this size and concatenates the results."
        ),
    )
    context_window: int = Field(
        default=8192,
        ge=1,
        description=(
            "Maximum tokens accepted in a single input text. Inputs beyond "
            "this are truncated or rejected by the backend, so passage chunks "
            "are sized against it. Defaults to 8192, matching the ~8k cap of "
            "most hosted models (e.g. OpenAI's text-embedding-3-*)."
        ),
    )
