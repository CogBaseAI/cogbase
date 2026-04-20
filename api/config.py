"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


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


class StructuredStoreConfig(BaseModel):
    type: Literal["sqlite", "postgres", "memory"] = "memory"
    path: str | None = None  # sqlite only
    url: str | None = None   # postgres only

    @model_validator(mode="after")
    def _validate(self) -> "StructuredStoreConfig":
        if self.type == "sqlite" and not self.path:
            raise ValueError("structured_store.path is required for sqlite type")
        if self.type == "postgres" and not self.url:
            raise ValueError("structured_store.url is required for postgres type")
        return self


class VectorStoreConfig(BaseModel):
    type: Literal["faiss", "pgvector"] = "faiss"
    dim: int = 1536
    url: str | None = None  # pgvector only

    @model_validator(mode="after")
    def _validate(self) -> "VectorStoreConfig":
        if self.type == "pgvector" and not self.url:
            raise ValueError("vector_store.url is required for pgvector type")
        return self


class ChunkerConfig(BaseModel):
    type: Literal["fixed", "langchain"] = "fixed"
    chunk_size: int = 512
    overlap: int = 64


class PackConfig(BaseModel):
    name: str  # e.g. "legal.contract_analyst"


class AppConfig(BaseModel):
    name: str
    llm: LLMConfig
    embedding: EmbeddingConfig | None = None
    structured_store: StructuredStoreConfig | None = None
    vector_store: VectorStoreConfig | None = None
    chunker: ChunkerConfig | None = None
    pack: PackConfig | None = None

    @model_validator(mode="after")
    def _validate_vector_config(self) -> "AppConfig":
        # embedding and chunker must be provided together.
        emb_chk = sum(x is not None for x in (self.embedding, self.chunker))
        if 0 < emb_chk < 2:
            raise ValueError("embedding and chunker must both be provided or both omitted")
        # An explicit vector_store requires embedding + chunker in the same config.
        if self.vector_store is not None and emb_chk == 0:
            raise ValueError(
                "vector_store requires embedding and chunker to also be provided"
            )
        return self

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "AppConfig":
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("YAML must be a mapping at the top level")
        return cls.model_validate(data)
