"""Request and response models for the CogBase REST API."""

from __future__ import annotations

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


class TaskResponse(BaseModel):
    task_id: str
    app_name: str
    task_type: str
    task_name: str
    doc_id: str | None
    params_json: str | None
    status: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


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
    user_id: str | None = Field(
        default=None,
        description="Optional user identifier associated with the session (memory scoping).",
    )


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


class QueryResponse(BaseModel):
    answer: str
    structured_records: list[dict] = []
    chunks: list[ChunkResponse] = []
    document_slices: list[DocumentSliceResponse] = []
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
    skill_id: str


class AppSkillRef(BaseModel):
    skill_id: str
    name: str | None = None  # display name; None if the referenced skill is missing


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
