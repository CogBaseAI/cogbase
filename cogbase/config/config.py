"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.config.models import LLMConfig, EmbeddingConfig


class ChunkerConfig(BaseModel):
    type: Literal["fixed", "langchain"] = "fixed"
    chunk_size: int = 512
    overlap: int = 64


class ChunkCollectionConfig(BaseModel):
    name: str
    chunker: ChunkerConfig
    dimensions: int = 1536
    description: str = "Full-text passage chunks; use for detailed or specific questions about document content."


class ExtractorConfig(BaseModel):
    type: Literal["llm"] = "llm"
    prompt: str | None = None
    extract_as_list: bool = False
    list_field: str = "items"
    item_id_field: str = "item_id"


class StructuredCollectionConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    schema_: str = Field(alias="schema")
    extractor: ExtractorConfig
    description: str = "Extracted structured records from ingested documents."


class DocumentCollectionConfig(BaseModel):
    name: str
    prompt: str | None = None
    dimensions: int = 1536
    description: str = "One-per-document summaries; use for topic-level or high-level questions about what documents cover."
    max_tokens: int = 1024
    metadata_fields: list[str] = []


class WhenCondition(BaseModel):
    metadata: dict[str, str] = {}


class PipelineStepConfig(BaseModel):
    tool: Literal["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
    collection: str
    when: WhenCondition | None = None


class PipelineConfig(BaseModel):
    parallel: bool = True
    steps: list[PipelineStepConfig] = []


# ---------------------------------------------------------------------------
# Workflow config
# ---------------------------------------------------------------------------


class WorkflowTriggerConfig(BaseModel):
    type: Literal["manual", "after_ingest"] = "manual"
    when: WhenCondition | None = None


class WorkflowOutputCollectionConfig(BaseModel):
    """A structured collection created by the workflow factory (no extractor)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    schema_: str = Field(alias="schema")
    primary_fields: list[str] = []
    description: str = ""


class WorkflowStepConfig(BaseModel):
    """One step in a workflow — either a leaf tool call or a foreach loop."""

    id: str
    # Leaf step
    tool: Literal["structured-query", "vector-search", "llm-structured", "structured-save"] | None = None
    # Foreach loop (mutually exclusive with tool)
    foreach: str | None = None
    steps: list["WorkflowStepConfig"] | None = None

    # structured-query / structured-save
    collection: str | None = None
    filters: dict[str, str] = {}

    # vector-search
    query: str | None = None
    top_k: int = 5

    # llm-structured
    prompt: str | None = None
    input: dict[str, Any] = {}
    output_schema: str | None = None  # JSON schema content (resolved from file ref)

    # structured-save
    records: list[Any] = []


WorkflowStepConfig.model_rebuild()


class WorkflowConfig(BaseModel):
    name: str
    trigger: WorkflowTriggerConfig = WorkflowTriggerConfig()
    input_schema: dict[str, str] = {}
    output_collections: list[WorkflowOutputCollectionConfig] = []
    steps: list[WorkflowStepConfig] = []


class AppConfig(BaseModel):
    name: str
    llm: LLMConfig | None = None
    embedding: EmbeddingConfig | None = None
    document_store: DocumentStoreConfig | None = None
    structured_store: StructuredStoreConfig | None = None
    vector_store: VectorStoreConfig | None = None
    chunk_collections: list[ChunkCollectionConfig] = []
    structured_collections: list[StructuredCollectionConfig] = []
    document_collections: list[DocumentCollectionConfig] = []
    pipeline: PipelineConfig | None = None
    skills: list[str] = []
    workflows: list[WorkflowConfig] = []

    @model_validator(mode="after")
    def _validate(self) -> "AppConfig":
        if self.chunk_collections and self.embedding is None:
            raise ValueError("embedding is required when chunk_collections are defined")
        if self.document_collections and self.embedding is None:
            raise ValueError("embedding is required when document_collections are defined")
        if self.pipeline:
            vc_names = {vc.name for vc in self.chunk_collections}
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
