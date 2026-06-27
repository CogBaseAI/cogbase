"""Request and response models for the CogBase REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from api.system_store import DocWorkflowStatus, TaskStatus


class ApplicationResponse(BaseModel):
    name: str
    status: str   # "initializing" | "active" | "error"
    config: dict[str, Any]
    error: str | None
    created_at: str
    updated_at: str


class ApplicationListResponse(BaseModel):
    applications: list[ApplicationResponse]
    total: int


# ---------------------------------------------------------------------------
# Doc registry models
# ---------------------------------------------------------------------------


class DocResponse(BaseModel):
    doc_id: str
    app_name: str
    status: str   # "active" | "failed" | "deleted"
    ingested_at: str
    metadata: dict[str, Any] = {}


class DocListResponse(BaseModel):
    docs: list[DocResponse]
    total: int


# ---------------------------------------------------------------------------
# Ingest models
# ---------------------------------------------------------------------------


class IngestDocumentsAcceptedResponse(BaseModel):
    task_ids: list[str]
    total: int
    batch_id: str = Field(
        description="Id grouping this upload's tasks; pass to GET /tasks/summary to track the batch."
    )


class IngestResultSummary(BaseModel):
    """Per-document outcome of a finished ingest task."""

    chunks_written: int = 0
    records_extracted: int = 0
    warning: str | None = Field(
        default=None,
        description=(
            "Set when the document ingested successfully but produced nothing — "
            "e.g. a scanned/image-only PDF with no extractable text, or content no "
            "pipeline step captured. The task still reports 'done'."
        ),
    )


class TaskResponse(BaseModel):
    task_id: str
    app_name: str
    task_type: str
    task_name: str
    doc_id: str | None
    batch_id: str | None = None
    params_json: str | None
    status: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    result: IngestResultSummary | None = Field(
        default=None, description="Ingest counts and any warning, once the task has finished."
    )


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


class TaskSummaryResponse(BaseModel):
    """Rollup of a set of background tasks — answers 'did my upload work?'."""

    app_name: str
    batch_id: str | None = None
    total: int
    pending: int
    running: int
    done: int
    failed: int
    chunks_written: int = Field(description="Total vector chunks written across finished ingest tasks.")
    records_extracted: int = Field(description="Total structured records written across finished ingest tasks.")
    warnings: int = Field(description="Number of finished ingest tasks that ingested nothing.")


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    text: str
    history: list[ChatMessage] = []
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Optional system prompt for this request. "
            "When set, overrides the app-level query_prompt configured in the application's "
            "config.yaml. Useful for prompt experimentation. You can expirement prompts and "
            "and set the final prompt into the app config for production."
        ),
    )
    top_k: int = Field(
        default=10,
        description=(
            "Default number of chunks returned per vector_search call. "
            "The LLM may request fewer; this caps and defaults its top_k argument. "
            "Hard upper limit is 20."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional short-term memory session id. When set, the server records "
            "this turn into the session and builds the LLM context from prior turns "
            "in that session, so `history` need not be supplied. Reuse the same id "
            "across requests to continue a conversation. Omit for stateless queries."
        ),
    )


class SessionStartRequest(BaseModel):
    metadata: dict | None = Field(
        default=None, description="Arbitrary session metadata seeded into the short-term cache."
    )
    session_id: str | None = Field(
        default=None, description="Resume an existing session id instead of creating a new one."
    )


class SessionResponse(BaseModel):
    session_id: str


class SessionCloseResponse(BaseModel):
    session_id: str
    distillation: str = Field(
        description="One of 'enqueued' / 'skipped' — whether a distillation task was started on close."
    )
    task_id: str | None = Field(
        default=None, description="The distillation task id when one was enqueued."
    )


class MemoryRecordResponse(BaseModel):
    """A long-term memory record surfaced to a reviewer.

    Includes the provenance (``source_event_ids`` / ``evidence_snapshot``) so a
    reviewer can audit the evidence before promoting a gated record to active.
    """

    memory_id: str
    kind: str
    content: str
    entities: list[str] = []
    confidence: float
    status: str
    source_event_ids: list[dict] = []
    evidence_snapshot: dict = {}
    observed_at: datetime
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class PendingMemoriesResponse(BaseModel):
    memories: list[MemoryRecordResponse]


class MemoryListResponse(BaseModel):
    """A page of stored long-term memories for the inspection surface."""

    memories: list[MemoryRecordResponse]
    total: int = Field(description="Number of records on this page.")


class MemoryReviewItem(BaseModel):
    memory_id: str
    decision: Literal["accept", "reject"] = Field(
        description="'accept' promotes the gated record to active; 'reject' marks it superseded."
    )


class MemoryReviewRequest(BaseModel):
    decisions: list[MemoryReviewItem] = Field(
        description="Per-record verdicts applied in one batch (server-capped)."
    )


class MemoryReviewResultItem(BaseModel):
    memory_id: str
    outcome: str = Field(
        description="One of 'accepted' / 'rejected' / 'skipped' (not pending) / 'not_found'."
    )


class MemoryReviewResponse(BaseModel):
    results: list[MemoryReviewResultItem]


class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = {}
    char_offset: int | None = None
    char_length: int | None = None


class DocumentSliceResponse(BaseModel):
    doc_id: str
    offset: int
    length: int
    text: str


class QueryMemoryResponse(BaseModel):
    """A long-term memory the answer drew on.

    A query-facing projection of ``LongTermRecord`` — only the fields useful for
    explaining the answer, without the reviewer-facing provenance carried by
    ``MemoryRecordResponse``.
    """

    memory_id: str
    kind: str
    content: str
    entities: list[str] = []


class AddMemoryMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        description="Speaker role; maps to the episodic continuity thread the distiller reads."
    )
    content: str = Field(description="The message text.")


class AddMemoryRequest(BaseModel):
    """Add a batch of conversation messages to long-term memory (mem0 ``add`` shape)."""

    messages: list[AddMemoryMessage] = Field(
        description="Conversation messages to distill into durable memories, in order."
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session to append to; a fresh one is generated and returned "
            "when omitted, so each call is an isolated, independently-distilled session."
        ),
    )
    metadata: dict | None = Field(
        default=None, description="Arbitrary session metadata seeded onto the session."
    )
    observation_date: datetime | None = Field(
        default=None,
        description=(
            "When the conversation took place; pins relative time references so they "
            "resolve correctly at distill time. Defaults to now."
        ),
    )


class AddMemoryResponse(BaseModel):
    session_id: str = Field(description="The session the messages were appended to.")
    memories: list[QueryMemoryResponse] = Field(
        default=[],
        description="The long-term memories this call created or reinforced (now active).",
    )


class QueryResponse(BaseModel):
    answer: str
    structured_records: list[dict] = []
    chunks: list[ChunkResponse] = []
    document_slices: list[DocumentSliceResponse] = []
    memories: list[QueryMemoryResponse] = []
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: str | None = Field(
        default=None,
        description="The short-term memory session id used for this turn, when memory was engaged.",
    )


# ---------------------------------------------------------------------------
# Skill models
# ---------------------------------------------------------------------------


class SkillResponse(BaseModel):
    id: str
    name: str
    description: str
    metadata: dict[str, Any] = {}
    source_path: str | None = None
    builtin: bool = False


class SkillListResponse(BaseModel):
    skills: list[SkillResponse]
    total: int


class AddSkillRequest(BaseModel):
    skill_name: str


class AppSkillRef(BaseModel):
    name: str  # display name; a referenced skill always exists (it can't be deleted while referenced)


class AppSkillsResponse(BaseModel):
    app_name: str
    skills: list[AppSkillRef]


# ---------------------------------------------------------------------------
# Collections / structured store models
# ---------------------------------------------------------------------------


class CollectionsResponse(BaseModel):
    structured: list[str]
    vector: list[str]


class FilterRequest(BaseModel):
    field: str
    op: str
    value: Any = None


class CollectionQueryRequest(BaseModel):
    filters: list[FilterRequest] = []
    fields: list[str] | None = None


class CollectionQueryResponse(BaseModel):
    collection: str
    records: list[dict]
    total: int


# ---------------------------------------------------------------------------
# Workflow models
# ---------------------------------------------------------------------------


class WorkflowListResponse(BaseModel):
    app_name: str
    workflows: list[str]


class DocWorkflowResponse(DocResponse):
    workflow_status: DocWorkflowStatus


class WorkflowDocListResponse(BaseModel):
    app_name: str
    workflow_name: str
    docs: list[DocWorkflowResponse]
    total: int


class WorkflowRunRequest(BaseModel):
    doc_id: str | None = None



# ---------------------------------------------------------------------------
# Generator models
# ---------------------------------------------------------------------------


class GenerateChatRequest(BaseModel):
    text: str
    history: list[ChatMessage] = []


class GenerateChatResponse(BaseModel):
    content: str        # display text (CONFIG markers stripped); store full in history
    config_yaml: str | None = None


class GenerateDeployRequest(BaseModel):
    config_yaml: str


class DeployResponse(BaseModel):
    name: str
    status: str
    error: str | None = None


# ---------------------------------------------------------------------------
# System config models
# ---------------------------------------------------------------------------


class SystemLLMConfigResponse(BaseModel):
    provider: str
    base_url: str
    api_key: str
    model: str
    mini_model: str | None = None


class SystemEmbeddingConfigResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key: str
    dimensions: int


class SystemConfigResponse(BaseModel):
    llm: SystemLLMConfigResponse | None = None
    embedding: SystemEmbeddingConfigResponse | None = None


class UpdateLLMConfig(BaseModel):
    provider: Literal["openai", "openai-compatible"] = "openai"
    model: str
    mini_model: str | None = None
    base_url: str = 'https://api.openai.com/v1'
    api_key: str = Field(
        description="API key. Use 'EMPTY' for local openai-compatible servers that require no auth (e.g. vLLM).",
    )


class UpdateEmbeddingConfig(BaseModel):
    provider: Literal["openai", "openai-compatible"] = "openai"
    model: str
    base_url: str = 'https://api.openai.com/v1'
    api_key: str = Field(
        description="API key. Use 'EMPTY' for local openai-compatible servers that require no auth (e.g. vLLM).",
    )
    dimensions: int


class UpdateSystemConfigRequest(BaseModel):
    llm: UpdateLLMConfig | None = None
    embedding: UpdateEmbeddingConfig | None = None
