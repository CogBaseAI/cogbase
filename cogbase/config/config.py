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
    # skip metadata_fields for app generator, most end users will not have additional
    # metadata for the documents. This is only used by developers.
    metadata_fields: list[str] = Field(
        default_factory=list,
        description="Metadata keys copied onto each stored vector.",
        json_schema_extra={"prompt_skip": True},
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
    steps: list[PipelineStepConfig] = Field(
        description="List of supported pipeline steps. Add the steps required for the application.",
    )


# ---------------------------------------------------------------------------
# Workflow config
# ---------------------------------------------------------------------------


class WorkflowParamsFromCollectionConfig(ConfigPromptMixin, BaseModel):
    collection: str = Field(
        description="Structured collection to query for deriving workflow params.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Equality filters rendered against the document context. "
            "Use templates such as '{{ doc.doc_id }}'."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Workflow params rendered for each matching record. "
            "Templates can reference '{{ doc }}' and '{{ record }}'."
        ),
    )
    distinct: bool = Field(
        default=True,
        description="Whether duplicate rendered param sets should be collapsed.",
    )


class WorkflowTriggerConfig(ConfigPromptMixin, BaseModel):
    type: Literal["manual", "after_ingest"] = Field(
        default="manual",
        description="Workflow trigger type.",
    )
    when: WhenCondition | None = Field(
        default=None,
        description="Optional condition that must match before triggering.",
    )


class WorkflowStepBase(ConfigPromptMixin, BaseModel):
    id: str = Field(
        description="Unique step identifier. Other steps reference this step's output via {{ steps.<id> }}.",
    )


class StructuredQueryStepConfig(WorkflowStepBase):
    tool: Literal["structured-query"] = Field(
        default="structured-query",
        description="Query a structured collection with optional equality filters and return matching records.",
    )
    collection: str = Field(
        description="Name of the structured collection to query.",
    )
    filters: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Equality filters applied to the query. Keys are field names; values are "
            "Jinja2 templates rendered against the step context "
            "(e.g. '{{ input.doc_id }}', '{{ item.clause_type }}')."
        ),
    )


class VectorSearchStepConfig(WorkflowStepBase):
    tool: Literal["vector-search"] = Field(
        default="vector-search",
        description="Embed a query and return the closest passages from a vector collection.",
    )
    collection: str = Field(
        description="Name of the vector collection to search.",
    )
    query: str = Field(
        description=(
            "Query text to embed and search with. Supports Jinja2 templates rendered "
            "against the step context (e.g. '{{ item.clause_type }}\\n{{ item.text }}')."
        ),
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of vector matches to return.",
    )


class LLMStructuredStepConfig(WorkflowStepBase):
    tool: Literal["llm-structured"] = Field(
        default="llm-structured",
        description=(
            "Call an LLM with a system prompt and structured input values, "
            "then parse its response against a JSON schema."
        ),
    )
    prompt: str = Field(
        description=(
            "System prompt sent to the LLM. Supports Jinja2 templates rendered "
            "against the step context."
        ),
    )
    input: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Named input values serialized as JSON and appended to the LLM user message. "
            "Each value is a Jinja2 template rendered against the step context "
            "(e.g. '{{ item }}', '{{ steps.load.records }}')."
        ),
    )
    output_schema: str = Field(
        description=(
            "JSON schema string the LLM response must conform to. "
            "The parsed result is stored as {{ steps.<id>.output }}."
        ),
    )


class StructuredSaveStepConfig(WorkflowStepBase):
    tool: Literal["structured-save"] = Field(
        default="structured-save",
        description="Render and upsert one or more records into a structured collection.",
    )
    collection: str = Field(
        description="Name of the structured collection to save into.",
    )
    records: list[Any] = Field(
        default_factory=list,
        description=(
            "List of records to save. Each entry is a Jinja2 template that resolves to "
            "a Pydantic model or dict (e.g. '{{ steps.judge.output }}')."
        ),
    )


WorkflowLeafStepConfig = Annotated[
    StructuredQueryStepConfig | VectorSearchStepConfig | LLMStructuredStepConfig | StructuredSaveStepConfig,
    Field(discriminator="tool"),
]


class ForeachStepConfig(WorkflowStepBase):
    foreach: str = Field(
        description=(
            "Jinja2 expression that resolves to a list to iterate over "
            "(e.g. '{{ steps.load.records }}', '{{ input.items }}')."
        ),
    )
    steps: list["WorkflowStepConfig"] = Field(
        description=(
            "Steps executed for every item in the foreach list. "
            "Each iteration exposes the current item as {{ item }}. "
            "Inner steps can also reference outer step outputs via {{ steps.<id> }}."
        ),
    )


WorkflowStepConfig = WorkflowLeafStepConfig | ForeachStepConfig

ForeachStepConfig.model_rebuild()


class WorkflowConfig(ConfigPromptMixin, BaseModel):
    name: str = Field(description="Workflow name.")
    trigger: WorkflowTriggerConfig = Field(
        default_factory=WorkflowTriggerConfig,
        description="Workflow trigger configuration.",
    )
    params_from_collection: WorkflowParamsFromCollectionConfig = Field(
        description=(
            "How to derive workflow params from a document. "
            "Queries a structured collection by doc_id and fans out one run per "
            "distinct rendered param set. Drives both after_ingest triggers and "
            "manual /run and /stream calls (caller passes doc_id)."
        ),
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
    name: str = Field(
        description="Application name, kebab-case (lowercase, alphanumeric, hyphens only).",
    )
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
