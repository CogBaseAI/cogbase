"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str
    api_key: str | None = None

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get("OPENAI_API_KEY")


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "sentence-transformers"] = "openai"
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    dimensions: int | None = None
