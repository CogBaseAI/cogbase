"""Request and response models for the CogBase REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


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


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    text: str


class QueryResponse(BaseModel):
    answer: str
    passthrough: bool = False
    structured_records: list[dict] = []


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
