"""Pydantic models for parsing application YAML configs."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.config.models import LLMConfig, EmbeddingConfig


class RecordMode(str, Enum):
    ONE = "one"
    MANY = "many"


class ChunkerConfig(BaseModel):
    type: Literal["fixed", "langchain"] = "langchain"
    chunk_size: int = 1024
    overlap: int = 128


class VectorCollectionConfig(BaseModel):
    name: str
    dimensions: int = 1536
    description: str
    metadata_fields: list[str] = []

    @model_validator(mode="after")
    def _non_empty_description(self) -> "VectorCollectionConfig":
        if not self.description or not self.description.strip():
            raise ValueError("VectorCollectionConfig.description must be set")
        return self


class ExtractorConfig(BaseModel):
    type: Literal["llm"] = "llm"
    extraction_schema: str
    prompt: str | None = None
    record_mode: RecordMode = RecordMode.ONE
    response_field: str = "items"
    id_field: str | None = None
    id_template: str | None = None


class StructuredCollectionConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    schema_: str = Field(alias="schema")
    primary_fields: list[str] = []
    description: str

    @model_validator(mode="after")
    def _non_empty_description(self) -> "StructuredCollectionConfig":
        if not self.description or not self.description.strip():
            raise ValueError("StructuredCollectionConfig.description must be set")
        return self


class WhenCondition(BaseModel):
    metadata: dict[str, str] = {}


class PipelineStepConfig(BaseModel):
    tool: Literal["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
    collection: str
    chunker: ChunkerConfig | None = None
    extractor: ExtractorConfig | None = None
    doc_prompt: str | None = Field(
        default=None,
        description="System instructions for the document level summarization LLM. Be specific."
    )


class PipelineConfig(BaseModel):
    name: str
    match: WhenCondition | None = None
    parallel: bool = False
    steps: list[PipelineStepConfig]


# ---------------------------------------------------------------------------
# Workflow config
# ---------------------------------------------------------------------------


class WorkflowTriggerConfig(BaseModel):
    type: Literal["manual", "after_ingest"] = "manual"
    when: WhenCondition | None = None


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
    steps: list[WorkflowStepConfig] = []


class AppConfig(BaseModel):
    name: str
    llm: LLMConfig | None = None
    embedding: EmbeddingConfig | None = None
    document_store: DocumentStoreConfig | None = None
    structured_store: StructuredStoreConfig | None = None
    vector_store: VectorStoreConfig | None = None
    vector_collections: list[VectorCollectionConfig] = []
    structured_collections: list[StructuredCollectionConfig] = []
    pipelines: list[PipelineConfig] = []
    skills: list[str] = []
    workflows: list[WorkflowConfig] = []

    @model_validator(mode="after")
    def _validate(self) -> "AppConfig":
        vc_names = {vc.name for vc in self.vector_collections}
        sc_names = {sc.name for sc in self.structured_collections}
        for pipeline in self.pipelines:
            for step in pipeline.steps:
                if step.tool in ("chunk-embed-upsert", "document-embed-upsert") and step.collection not in vc_names:
                    raise ValueError(
                        f"Pipeline step references unknown vector collection: {step.collection!r}"
                    )
                if step.tool == "extract-structured" and step.collection not in sc_names:
                    raise ValueError(
                        f"Pipeline step references unknown structured collection: {step.collection!r}"
                    )
                if step.tool == "extract-structured" and step.extractor is None:
                    raise ValueError(
                        f"Pipeline step for {step.collection!r} is missing 'extractor'"
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
        from typing import get_args

        tools = get_args(PipelineStepConfig.model_fields["tool"].annotation)
        chunk_tool, extract_tool, doc_tool = tools
        return (
            "name: <kebab-case-name>\n"
            "\n"
            "vector_collections:\n"
            "  - name: <snake_case>\n"
            '    description: "<shown to the LLM as context during retrieval>"\n'
            "\n"
            "structured_collections:\n"
            "  - name: <snake_case>\n"
            '    description: "<shown to the LLM as context during lookup>"\n'
            "    schema: '<record_schema JSON string from ---SCHEMA RESOLVED--->'\n"
            "    primary_fields: [doc_id]\n"
            "\n"
            "pipelines:\n"
            "  - name: <name>\n"
            "    steps:\n"
            f"      - tool: {chunk_tool}\n"
            "        collection: <vector_collection>\n"
            "        chunker:\n"
            "          type: langchain\n"
            "\n"
            f"      - tool: {extract_tool}          # include only if structured extraction is needed\n"
            "        collection: <structured_collection>\n"
            "        extractor:\n"
            "          type: llm\n"
            "          extraction_schema: '<extraction_schema JSON string from ---SCHEMA RESOLVED--->'\n"
            "          prompt: |\n"
            "            <System instructions for the extraction LLM. Be specific.>\n"
            "\n"
            f"      - tool: {doc_tool}       # include only if summary/topic queries are needed\n"
            "        collection: <vector_collection>\n"
            "        doc_prompt: |\n"
            "          <System instructions for the document level summarization LLM. Be specific.>\n"
        )
