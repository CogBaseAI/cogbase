"""Request and response models for the CogBase REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
# Ingest models
# ---------------------------------------------------------------------------


class DocumentRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any] = {}


class IngestDocumentsRequest(BaseModel):
    documents: list[DocumentRequest]
    concurrency: int = 5


class IngestResultResponse(BaseModel):
    doc_id: str
    success: bool
    records_extracted: int
    error: str | None


class IngestDocumentsResponse(BaseModel):
    results: list[IngestResultResponse]


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
    started_at: str
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


class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = {}
    char_offset: int | None = None
    char_length: int | None = None


class QueryResponse(BaseModel):
    answer: str
    structured_records: list[dict] = []
    chunks: list[ChunkResponse] = []


# ---------------------------------------------------------------------------
# Skill models
# ---------------------------------------------------------------------------


class SkillResponse(BaseModel):
    name: str
    description: str
    metadata: dict[str, Any] = {}
    source_path: str | None = None


class SkillListResponse(BaseModel):
    skills: list[SkillResponse]
    total: int


class AddSkillRequest(BaseModel):
    skill_name: str


class AppSkillsResponse(BaseModel):
    app_name: str
    skills: list[str]


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


class WorkflowRunRequest(BaseModel):
    doc_id: str


class WorkflowRunResponse(BaseModel):
    workflow: str
    records: list[dict]
    total: int


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
