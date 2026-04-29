"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.config.models import LLMConfig, EmbeddingConfig


class ChunkerConfig(BaseModel):
    type: Literal["fixed", "langchain"] = "fixed"
    chunk_size: int = 512
    overlap: int = 64


class VectorCollectionConfig(BaseModel):
    name: str
    chunker: ChunkerConfig
    description: str = ""


class ExtractorConfig(BaseModel):
    type: Literal["llm"] = "llm"
    prompt: str | None = None


class StructuredCollectionConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    schema_: str = Field(alias="schema")
    extractor: ExtractorConfig


class DocumentCollectionConfig(BaseModel):
    name: str
    description: str = ""
    prompt: str | None = None
    max_tokens: int = 1024
    metadata_fields: list[str] = []


class PipelineStepConfig(BaseModel):
    tool: Literal["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
    collection: str


class PipelineConfig(BaseModel):
    parallel: bool = True
    steps: list[PipelineStepConfig] = []


class AppConfig(BaseModel):
    name: str
    llm: LLMConfig
    embedding: EmbeddingConfig | None = None
    document_store: DocumentStoreConfig | None = None
    structured_store: StructuredStoreConfig | None = None
    vector_store: VectorStoreConfig | None = None
    vector_collections: list[VectorCollectionConfig] = []
    structured_collections: list[StructuredCollectionConfig] = []
    document_collections: list[DocumentCollectionConfig] = []
    pipeline: PipelineConfig | None = None
    skills: list[str] = []

    @model_validator(mode="after")
    def _validate(self) -> "AppConfig":
        if self.vector_collections and self.embedding is None:
            raise ValueError("embedding is required when vector_collections are defined")
        if self.document_collections and self.embedding is None:
            raise ValueError("embedding is required when document_collections are defined")
        if self.pipeline:
            vc_names = {vc.name for vc in self.vector_collections}
            sc_names = {sc.name for sc in self.structured_collections}
            dc_names = {dc.name for dc in self.document_collections}
            for step in self.pipeline.steps:
                if step.tool == "chunk-embed-upsert" and step.collection not in vc_names:
                    raise ValueError(
                        f"Pipeline step references unknown vector collection: {step.collection!r}"
                    )
                if step.tool == "extract-structured" and step.collection not in sc_names:
                    raise ValueError(
                        f"Pipeline step references unknown structured collection: {step.collection!r}"
                    )
                if step.tool == "document-embed-upsert" and step.collection not in dc_names:
                    raise ValueError(
                        f"Pipeline step references unknown document collection: {step.collection!r}"
                    )
        return self

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "AppConfig":
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("YAML must be a mapping at the top level")
        return cls.model_validate(data)
