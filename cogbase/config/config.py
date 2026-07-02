"""Pydantic models for parsing application YAML configs.

Schema lifecycle
----------------
Pipeline extraction schema  (ExtractorConfig.extraction_schema)
    The JSON schema the LLM receives during ingest.  It covers only the fields
    the LLM should extract from the document text — never ``doc_id`` and never
    the ``id_field`` used by RecordMode.MANY (e.g. ``clause_id``).  Those
    identifiers are injected automatically at ingest time and must not appear in
    the extraction schema.

Structured-collection record schema  (StructuredCollectionConfig.schema_)
    The full schema stored in the collection.  Built from the extraction schema
    plus fields injected at config-load time:
      • ``doc_id``   — always present; links every record to its source document.
      • ``id_field`` — present only for RecordMode.MANY (e.g. ``clause_id``);
                       generated from ``id_template`` rather than extracted.

    This is the schema that callers and workflows work with.

Workflows operate on record schemas, not extraction schemas
    ``structured-query`` returns records shaped by the collection's record
    schema.  ``structured-save`` writes records whose fields must satisfy that
    same schema.  When a workflow needs to save a finding keyed by both
    ``doc_id`` and ``clause_id`` (e.g. clause compliance findings), both fields
    must appear in the relevant ``llm-structured`` step's ``output_schema``,
    because the LLM receives them as input and must pass them through verbatim.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Any, Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.config.models import LLMConfig, EmbeddingConfig
from cogbase.config.prompt import ConfigPromptMixin, render_config_template
from cogbase.stores.schema import validate_resource_name


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
    # Extraction-only schema: no doc_id, no id_field.  See module docstring.
    extraction_schema: str = Field(description="Resolved JSON schema used for extraction.")
    prompt: str = Field(
        description="System prompt for the extraction LLM.",
    )
    record_mode: RecordMode = Field(
        default=RecordMode.ONE,
        description="Whether the extractor returns one record or many records for one document.",
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

    @model_validator(mode="after")
    def _validate_record_mode_contract(self) -> "ExtractorConfig":
        if self.record_mode == RecordMode.MANY:
            missing = [
                f for f, v in [
                    ("response_field", self.response_field),
                    ("id_field", self.id_field),
                    ("id_template", self.id_template),
                ]
                if not v
            ]
            if missing:
                raise ValueError(
                    f"record_mode=many requires: {', '.join(missing)}"
                )
            # Strip id_field from extraction_schema if the LLM included it — it is
            # injected automatically via id_template and must not be extracted.
            if self.id_field:
                try:
                    schema = json.loads(self.extraction_schema)
                except (json.JSONDecodeError, ValueError):
                    schema = None
                if schema and self.id_field in schema.get("properties", {}):
                    schema["properties"].pop(self.id_field)
                    if self.id_field in schema.get("required", []):
                        schema["required"].remove(self.id_field)
                    self.extraction_schema = json.dumps(schema, separators=(",", ":"))
        elif self.record_mode == RecordMode.ONE:
            if self.id_field is not None:
                raise ValueError("id_field must not be set for record_mode=one")
        return self


class StructuredCollectionConfig(ConfigPromptMixin, BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Collection name.")
    description: str = Field(
        description="Collection description, shown to the LLM as context for a query.",
    )
    # Full record schema (extraction fields + doc_id + id_field).  See module docstring.
    # Skipped in app-generator prompts; injected explicitly by the generator.
    schema_: str = Field(
        alias="schema",
        description="Resolved JSON schema for the collection.",
        json_schema_extra={"prompt_skip": True},
    )
    primary_fields: list[str] = Field(
        default_factory=list,
        description="Primary lookup fields for the collection.",
        json_schema_extra={"prompt_skip": True},
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
    # note: structured-save is to save the output of llm-structured, primary_fields MUST be in
    # LLMStructuredStepConfig.output_schema
    primary_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Primary lookup fields for the target structured collection — the "
            "stable identifier fields that uniquely key each saved record "
            "(e.g. [doc_id, clause_id] or [finding_id]). Copied onto the target "
            "collection's primary_fields at validation time."
        ),
    )
    records: list[Any] = Field(
        default_factory=list,
        description=(
            "List of records to save. Each entry is a Jinja2 template that resolves to "
            "a Pydantic model or dict (e.g. '{{ steps.judge.output }}')."
        ),
    )
    purge_by: list[str] = Field(
        default_factory=lambda: ["doc_id"],
        description=(
            "Doc-linkage fields on this collection that reference a source document. "
            "Before a workflow re-runs for a re-ingested doc, rows where ANY of these "
            "fields == that doc_id are deleted, so resolved findings disappear instead "
            "of orphaning. Multiple fields are OR-ed (e.g. [doc_a_id, doc_b_id] for a "
            "cross-document finding). Default [doc_id]. Set [] to disable purge for "
            "outputs that aren't doc-scoped."
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


def _iter_save_steps(steps: list) -> "list[StructuredSaveStepConfig]":
    """Depth-first collect every structured-save step, descending into foreach blocks."""
    out: list[StructuredSaveStepConfig] = []
    for step in steps:
        if isinstance(step, StructuredSaveStepConfig):
            out.append(step)
        elif isinstance(step, ForeachStepConfig):
            out.extend(_iter_save_steps(step.steps))
    return out


def _iter_all_leaf_steps(steps: list) -> "list[WorkflowLeafStepConfig]":
    """Depth-first collect every non-foreach workflow step."""
    out = []
    for step in steps:
        if isinstance(step, ForeachStepConfig):
            out.extend(_iter_all_leaf_steps(step.steps))
        else:
            out.append(step)
    return out


_STEP_OUTPUT_RE = re.compile(r"\{\{\s*steps\.([A-Za-z_][A-Za-z0-9_]*)\.output\s*\}\}")


def _index_llm_structured_steps(steps: list) -> "dict[str, LLMStructuredStepConfig]":
    """Map step id -> LLMStructuredStepConfig for every llm-structured step in the tree."""
    out: dict[str, LLMStructuredStepConfig] = {}
    for step in steps:
        if isinstance(step, LLMStructuredStepConfig):
            out[step.id] = step
        elif isinstance(step, ForeachStepConfig):
            out.update(_index_llm_structured_steps(step.steps))
    return out


def _validate_save_primary_fields(workflow: "WorkflowConfig") -> None:
    """structured-save primary_fields must be declared in the upstream llm-structured output_schema.

    For each structured-save step, find the llm-structured step whose
    ``{{ steps.<id>.output }}`` feeds ``records``, then assert every field in
    ``primary_fields`` appears as a property in that step's ``output_schema``.
    Records that don't reference an llm-structured step output are skipped.
    """
    llm_steps = _index_llm_structured_steps(workflow.steps)
    for save_step in _iter_save_steps(workflow.steps):
        if not save_step.primary_fields:
            continue
        for record in save_step.records:
            if not isinstance(record, str):
                continue
            match = _STEP_OUTPUT_RE.search(record)
            if not match:
                continue
            upstream = llm_steps.get(match.group(1))
            if upstream is None:
                continue
            try:
                schema = json.loads(upstream.output_schema)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Workflow {workflow.name!r} step {upstream.id!r}: "
                    f"output_schema is not valid JSON: {exc}"
                ) from exc
            properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
            missing = [f for f in save_step.primary_fields if f not in properties]
            if missing:
                raise ValueError(
                    f"Workflow {workflow.name!r} structured-save step {save_step.id!r}: "
                    f"primary_fields {missing} missing from upstream llm-structured step "
                    f"{upstream.id!r}.output_schema.properties "
                    f"(available: {sorted(properties.keys())}). "
                    "Add these fields to the llm-structured output_schema and instruct the "
                    "LLM prompt to copy them verbatim from its input."
                )


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


class MemoryConfig(ConfigPromptMixin, BaseModel):
    """Long-term memory tuning for one app (distillation + reconciliation).

    Advanced, opt-in knobs the app generator deliberately does not author — every
    field is ``prompt_skip`` so a user sets them manually in ``config.yaml`` when
    they want to scope or tune the memory pipeline; omitting the section keeps the
    generic defaults.
    """

    domain_fact_guidance: str | None = Field(
        default=None,
        description=(
            "Application-specific description of which subject-matter facts are "
            "durable, injected as an additive topic scope into the distiller's "
            "extraction prompt. Narrows what counts as a `fact`/`correction`; it "
            "cannot relax the rule that the user must be the fact's source."
        ),
        json_schema_extra={"prompt_skip": True},
    )
    existing_memory_limit: int = Field(
        default=10,
        description=(
            "How many related existing memories the distiller vector-recalls and "
            "injects into the extraction prompt as a reconcile + linking "
            "reference. 0 disables the lookup (blind extraction)."
        ),
        json_schema_extra={"prompt_skip": True},
    )
    auto_link_max_entity_ratio: float = Field(
        default=0.1,
        description=(
            "Deterministic link augmentation: after extraction, a new memory is "
            "auto-linked to an existing recalled memory when they share an entity "
            "that is discriminative — present in no more than this fraction of all "
            "active records. Ubiquitous entities (e.g. the speakers in a two-person "
            "dialogue, who appear in most records) exceed the ratio and are ignored, "
            "so the graph stays sparse instead of collapsing into a same-subject "
            "clique. 0 disables auto-linking and leaves edges to the LLM alone."
        ),
        json_schema_extra={"prompt_skip": True},
    )
    single_call_reconcile: bool = Field(
        default=True,
        description=(
            "When true, the distiller's single extraction call also decides each "
            "memory's reconcile op (ADD/UPDATE/DELETE/NOOP) against the front-loaded "
            "existing memories — one LLM call per session. When false, fall back to "
            "the auditable per-candidate reconcile (one extra LLM call per "
            "candidate). Requires existing_memory_limit > 0 to have memories to "
            "reconcile against."
        ),
        json_schema_extra={"prompt_skip": True},
    )
    reconcile_guidance: str | None = Field(
        default=None,
        description=(
            "Application-specific guidance injected as an additive domain block "
            "into the long-term reconcile prompt — domain judgement about when two "
            "observations are the same claim versus a genuine contradiction. It "
            "cannot change the ADD/UPDATE/DELETE/NOOP operation set or output."
        ),
        json_schema_extra={"prompt_skip": True},
    )
    recall_neighbors: int = Field(
        default=5,
        description=(
            "How many extra memories recall appends by following the memory graph "
            "one hop out from its relevance hits (linked context the vector query "
            "alone would miss). 0 disables neighborhood expansion."
        ),
        json_schema_extra={"prompt_skip": True},
    )
    enable_memory_lookup: bool = Field(
        default=False,
        description=(
            "Whether the query runner exposes the `memory_lookup` tool for on-demand "
            "long-term memory recall. When false (the default), the tool is withheld "
            "even if a long-term tier is configured; memory injected into context "
            "still applies. Set true to opt into on-demand recall."
        ),
        json_schema_extra={"prompt_skip": True},
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
        description="Structured collections available to pipelines and workflows.",
    )
    pipelines: list[PipelineConfig] = Field(
        default_factory=list,
        description="Configured ingestion pipelines.",
    )
    query_prompt: str | None = Field(
        default=None,
        description="System prompt used by the query runner when answering user questions. Replaces the default 'You are a helpful assistant.' base prompt.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Ids of system-wide skills to load (see GET /skills). Referenced by id so a skill rename does not break the app.",
        json_schema_extra={"prompt_skip": True}, # skip for now, add back when needed
    )
    workflows: list[WorkflowConfig] = Field(
        default_factory=list,
        description="Configured workflows.",
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Long-term memory tuning (distillation + reconciliation).",
        json_schema_extra={"prompt_skip": True},
    )

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        return validate_resource_name(v)

    @model_validator(mode="after")
    def _validate(self) -> "AppConfig":
        vc_names = {vc.name for vc in self.vector_collections}
        sc_by_name = {sc.name: sc for sc in self.structured_collections}

        # Validate pipeline collection references.
        for pipeline in self.pipelines:
            for step in pipeline.steps:
                if isinstance(step, (ChunkEmbedUpsertStepConfig, DocumentEmbedUpsertStepConfig)) and step.collection not in vc_names:
                    raise ValueError(
                        f"Pipeline {pipeline.name!r} step references unknown vector collection: {step.collection!r}"
                    )
                if isinstance(step, ExtractStructuredStepConfig) and step.collection not in sc_by_name:
                    raise ValueError(
                        f"Pipeline {pipeline.name!r} step references unknown structured collection: {step.collection!r}"
                    )

        # Reject ambiguous multi-extractor writes to the same structured collection.
        # Two extract-structured steps may share a collection only if their schema and
        # prompt are identical (same extractor, different routing condition).
        extractor_by_collection: dict[str, tuple[str, ExtractStructuredStepConfig]] = {}
        for pipeline in self.pipelines:
            for step in pipeline.steps:
                if not isinstance(step, ExtractStructuredStepConfig):
                    continue
                if step.collection not in extractor_by_collection:
                    extractor_by_collection[step.collection] = (pipeline.name, step)
                else:
                    first_pipeline, first_step = extractor_by_collection[step.collection]
                    if (
                        step.extractor.extraction_schema != first_step.extractor.extraction_schema
                        or step.extractor.prompt != first_step.extractor.prompt
                    ):
                        raise ValueError(
                            f"Structured collection {step.collection!r} is written by "
                            f"extract-structured steps with different schemas or prompts: "
                            f"pipeline {first_pipeline!r} and pipeline {pipeline.name!r}. "
                            "Each structured collection must have a single consistent extractor."
                        )

        # Validate workflow collection references and save-step primary_fields.
        for workflow in self.workflows:
            pfc_col = workflow.params_from_collection.collection
            if pfc_col not in sc_by_name:
                raise ValueError(
                    f"Workflow {workflow.name!r} params_from_collection references unknown "
                    f"structured collection: {pfc_col!r}"
                )
            for step in _iter_all_leaf_steps(workflow.steps):
                if isinstance(step, (StructuredQueryStepConfig, StructuredSaveStepConfig)):
                    if step.collection not in sc_by_name:
                        raise ValueError(
                            f"Workflow {workflow.name!r} step {step.id!r} references unknown "
                            f"structured collection: {step.collection!r}"
                        )
                elif isinstance(step, VectorSearchStepConfig):
                    if step.collection not in vc_names:
                        raise ValueError(
                            f"Workflow {workflow.name!r} step {step.id!r} references unknown "
                            f"vector collection: {step.collection!r}"
                        )
            _validate_save_primary_fields(workflow)
            for save_step in _iter_save_steps(workflow.steps):
                if not save_step.primary_fields:
                    continue
                target = sc_by_name.get(save_step.collection)
                if target is not None:
                    target.primary_fields = list(save_step.primary_fields)
        return self

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "AppConfig":
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("YAML must be a mapping at the top level")
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.dump(
            self.model_dump(by_alias=True, mode="json", exclude_none=True),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    @classmethod
    def config_format_prompt(cls) -> str:
        """YAML config template for LLM system prompts; derived from the live model."""
        return render_config_template(cls)
