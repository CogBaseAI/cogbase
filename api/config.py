"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class VectorCollectionConfig(BaseModel):
    name: str
    chunker: ChunkerConfig


class ExtractorConfig(BaseModel):
    type: Literal["llm"] = "llm"
    # Prompt prefix for the LLM system message; the JSON schema is appended
    # automatically.  When omitted LLMExtractor uses its built-in default.
    prompt: str | None = None  # resolved from filename ref at upload time


class StructuredCollectionConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    schema_: str = Field(alias="schema")  # JSON Schema string; resolved from filename ref at upload time
    extractor: ExtractorConfig


class PipelineStepConfig(BaseModel):
    tool: Literal["chunk-embed-upsert", "extract-structured"]
    collection: str       # must match a vector_collection or structured_collection name


class PipelineConfig(BaseModel):
    parallel: bool = True
    steps: list[PipelineStepConfig] = []


class AppConfig(BaseModel):
    name: str
    llm: LLMConfig
    embedding: EmbeddingConfig | None = None
    structured_store: StructuredStoreConfig | None = None
    vector_store: VectorStoreConfig | None = None
    vector_collections: list[VectorCollectionConfig] = []
    structured_collections: list[StructuredCollectionConfig] = []
    pipeline: PipelineConfig | None = None
    skills: list[str] = []

    @model_validator(mode="after")
    def _validate(self) -> "AppConfig":
        if self.vector_collections and self.embedding is None:
            raise ValueError("embedding is required when vector_collections are defined")
        if self.pipeline:
            vc_names = {vc.name for vc in self.vector_collections}
            sc_names = {sc.name for sc in self.structured_collections}
            for step in self.pipeline.steps:
                if step.tool == "chunk-embed-upsert" and step.collection not in vc_names:
                    raise ValueError(
                        f"Pipeline step references unknown vector collection: {step.collection!r}"
                    )
                if step.tool == "extract-structured" and step.collection not in sc_names:
                    raise ValueError(
                        f"Pipeline step references unknown structured collection: {step.collection!r}"
                    )
        return self

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "AppConfig":
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("YAML must be a mapping at the top level")
        return cls.model_validate(data)
