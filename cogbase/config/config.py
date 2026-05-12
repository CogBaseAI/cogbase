"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.config.models import LLMConfig, EmbeddingConfig
from cogbase.config.prompt import ConfigPromptMixin, render_config_template


class RecordMode(str, Enum):
    ONE = "one"
    MANY = "many"


class ChunkerConfig(ConfigPromptMixin, BaseModel):
    type: Literal["fixed", "langchain"] = Field(
        default="langchain",
        description="Chunking strategy."
    )
    chunk_size: int = Field(default=1024, description="Chunk size in characters.")
    overlap: int = Field(default=128, description="Chunk overlap in characters.")


class VectorCollectionConfig(ConfigPromptMixin, BaseModel):
    name: str = Field(description="Collection name.")
    dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensionality.",
        json_schema_extra={"prompt_skip": True},
    )
    description: str = Field(
        description="Collection description, shown to the LLM as context for a query.",
    )
    metadata_fields: list[str] = Field(
        default_factory=list,
        description="Metadata keys copied onto each stored vector.",
    )

    @model_validator(mode="after")
    def _non_empty_description(self) -> "VectorCollectionConfig":
        if not self.description or not self.description.strip():
            raise ValueError("VectorCollectionConfig.description must be set")
        return self


class ExtractorConfig(ConfigPromptMixin, BaseModel):
    type: Literal["llm"] = Field(default="llm", description="Extractor implementation.")
    extraction_schema: str = Field(description="Resolved JSON schema used for extraction.")
    prompt: str = Field(
        description="System prompt for the extraction LLM.",
    )
    record_mode: RecordMode = Field(
        default=RecordMode.ONE,
        description="Whether the extractor returns one record or many.",
    )
    response_field: str | None = Field(
        default="items",
        description="Top-level response field containing extracted records. Only used for RecordMode.MANY",
    )
    id_field: str | None = Field(
        default=None,
        description="Optional record identifier field name. Required for RecordMode.MANY.",
    )
    id_template: str | None = Field(
        default=None,
        description="Optional template for generated record ids. Required for RecordMode.MANY.",
    )


class StructuredCollectionConfig(ConfigPromptMixin, BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Collection name.")
    schema_: str = Field(alias="schema", description="Resolved JSON schema for the collection.")
    primary_fields: list[str] = Field(
        default_factory=list,
        description="Primary lookup fields for the collection.",
    )
    description: str = Field(
        description="Collection description, shown to the LLM as context for a query.",
    )

    @model_validator(mode="after")
    def _non_empty_description(self) -> "StructuredCollectionConfig":
        if not self.description or not self.description.strip():
            raise ValueError("StructuredCollectionConfig.description must be set")
        return self


class WhenCondition(ConfigPromptMixin, BaseModel):
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata key/value filters required for the condition.",
    )


class PipelineStepBase(ConfigPromptMixin, BaseModel):
    collection: str = Field(description="Target collection name.")


class ChunkEmbedUpsertStepConfig(PipelineStepBase):
    tool: Literal["chunk-embed-upsert"] = Field(
        default="chunk-embed-upsert",
        description="Pipeline tool to run.",
    )
    chunker: ChunkerConfig = Field(
        default_factory=ChunkerConfig,
        description="Chunking settings for chunk-embed-upsert steps.",
        json_schema_extra={"prompt_skip": True},
    )


class ExtractStructuredStepConfig(PipelineStepBase):
    tool: Literal["extract-structured"] = Field(
        default="extract-structured",
        description="Pipeline tool to run.",
    )
    extractor: ExtractorConfig = Field(
        description="Extraction settings for extract-structured steps.",
    )


class DocumentEmbedUpsertStepConfig(PipelineStepBase):
    tool: Literal["document-embed-upsert"] = Field(
        default="document-embed-upsert",
        description="Pipeline tool to run.",
    )
    doc_prompt: str = Field(
        default="Summarize this document in a concise way, focusing on the most important points and avoiding unnecessary detail.",
        description="System instructions for the document level summarization LLM.",
    )


PipelineStepConfig = Annotated[
    ChunkEmbedUpsertStepConfig | ExtractStructuredStepConfig | DocumentEmbedUpsertStepConfig,
    Field(discriminator="tool"),
]


class PipelineConfig(ConfigPromptMixin, BaseModel):
    name: str = Field(description="Pipeline name.")
    routing_description: str = Field(
        description=(
            "Human-readable description of which documents belong in this pipeline. "
            "Used by LLM routing (strategy: llm or auto) to classify documents into the correct pipeline."
        ),
    )
    match: WhenCondition | None = Field(
        default=None,
        description="Optional condition that selects which documents enter this pipeline.",
    )
    parallel: bool = Field(
        default=False,
        description="Whether pipeline steps may run in parallel.",
    )
    steps: list[PipelineStepConfig] = Field(description="List of supported pipeline steps. Add the steps required for the application.")


# ---------------------------------------------------------------------------
# Workflow config
# ---------------------------------------------------------------------------


class WorkflowTriggerConfig(ConfigPromptMixin, BaseModel):
    type: Literal["manual", "after_ingest"] = Field(
        default="manual",
        description="Workflow trigger type.",
    )
    when: WhenCondition | None = Field(
        default=None,
        description="Optional condition that must match before triggering.",
    )


class WorkflowStepConfig(ConfigPromptMixin, BaseModel):
    """One step in a workflow — either a leaf tool call or a foreach loop."""

    id: str = Field(description="Step identifier.")
    # Leaf step
    tool: Literal["structured-query", "vector-search", "llm-structured", "structured-save"] | None = Field(
        default=None,
        description="Leaf workflow tool to run.",
    )
    # Foreach loop (mutually exclusive with tool)
    foreach: str | None = Field(
        default=None,
        description="Collection or input path to iterate over for foreach loops.",
    )
    steps: list["WorkflowStepConfig"] | None = Field(
        default=None,
        description="Nested workflow steps for foreach loops.",
    )

    # structured-query / structured-save
    collection: str | None = Field(
        default=None,
        description="Collection name for structured-query or structured-save steps.",
    )
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Key/value filters for structured-query or structured-save steps.",
    )

    # vector-search
    query: str | None = Field(
        default=None,
        description="Search query for vector-search steps.",
    )
    top_k: int = Field(default=5, description="Maximum number of vector matches to return.")

    # llm-structured
    prompt: str | None = Field(
        default=None,
        description="Prompt for llm-structured steps.",
    )
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Input mapping passed to llm-structured steps.",
    )
    output_schema: str | None = Field(
        default=None,
        description="Resolved JSON schema content for llm-structured output.",
    )

    # structured-save
    records: list[Any] = Field(
        default_factory=list,
        description="Records to save for structured-save steps.",
    )


WorkflowStepConfig.model_rebuild()


class WorkflowConfig(ConfigPromptMixin, BaseModel):
    name: str = Field(description="Workflow name.")
    trigger: WorkflowTriggerConfig = Field(
        default_factory=WorkflowTriggerConfig,
        description="Workflow trigger configuration.",
    )
    input_schema: dict[str, str] = Field(
        default_factory=dict,
        description="Input schema mapping used by the workflow.",
    )
    steps: list[WorkflowStepConfig] = Field(
        default_factory=list,
        description="Ordered list of workflow steps.",
    )


class RoutingStrategy(str, Enum):
    AUTO = "auto"
    METADATA = "metadata"
    LLM = "llm"


class PipelineRoutingConfig(BaseModel):
    strategy: RoutingStrategy = Field(
        default=RoutingStrategy.AUTO,
        description=(
            "Pipeline routing strategy. "
            "'auto' (default) — try metadata first; fall back to LLM if no metadata match. "
            "'metadata' — match by document metadata key/value pairs. "
            "'llm' — always use LLM to classify the document into a pipeline."
        ),
    )


class AppConfig(ConfigPromptMixin, BaseModel):
    name: str = Field(description="Application name, kebab-case (lowercase, alphanumeric, hyphens only).")
    pipeline_routing: PipelineRoutingConfig = Field(
        default_factory=PipelineRoutingConfig,
        description="Pipeline routing configuration.",
        json_schema_extra={"prompt_skip": True},
    )
    llm: LLMConfig | None = Field(
        default=None,
        description="LLM configuration.",
        json_schema_extra={"prompt_skip": True},
    )
    embedding: EmbeddingConfig | None = Field(
        default=None,
        description="Embedding configuration.",
        json_schema_extra={"prompt_skip": True},
    )
    document_store: DocumentStoreConfig | None = Field(
        default=None,
        description="Document storage configuration.",
        json_schema_extra={"prompt_skip": True},
    )
    structured_store: StructuredStoreConfig | None = Field(
        default=None,
        description="Structured record storage configuration.",
        json_schema_extra={"prompt_skip": True},
    )
    vector_store: VectorStoreConfig | None = Field(
        default=None,
        description="Vector index storage configuration.",
        json_schema_extra={"prompt_skip": True},
    )
    vector_collections: list[VectorCollectionConfig] = Field(
        default_factory=list,
        description="Vector collections.",
    )
    structured_collections: list[StructuredCollectionConfig] = Field(
        default_factory=list,
        description="Structured collections available to pipelines.",
    )
    pipelines: list[PipelineConfig] = Field(
        default_factory=list,
        description="Configured ingestion pipelines.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Additional skill names to load.",
        json_schema_extra={"prompt_skip": True}, # skip for now, add back when needed
    )
    workflows: list[WorkflowConfig] = Field(
        default_factory=list,
        description="Configured workflows.",
        json_schema_extra={"prompt_skip": True}, # skip for now, add back when needed
    )

    @model_validator(mode="after")
    def _validate(self) -> "AppConfig":
        vc_names = {vc.name for vc in self.vector_collections}
        sc_names = {sc.name for sc in self.structured_collections}
        for pipeline in self.pipelines:
            for step in pipeline.steps:
                if isinstance(step, (ChunkEmbedUpsertStepConfig, DocumentEmbedUpsertStepConfig)) and step.collection not in vc_names:
                    raise ValueError(
                        f"Pipeline step references unknown vector collection: {step.collection!r}"
                    )
                if isinstance(step, ExtractStructuredStepConfig) and step.collection not in sc_names:
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

    @classmethod
    def config_format_prompt(cls) -> str:
        """YAML config template for LLM system prompts; derived from the live model."""
        return render_config_template(cls)
