"""Store configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from cogbase.config.prompt import ConfigPromptMixin


class StructuredStoreConfig(ConfigPromptMixin, BaseModel):
    type: Literal["sqlite", "postgres", "memory"] = Field(
        default="memory",
        description="Structured store backend."
    )
    path: str | None = Field(
        default=None,
        description="SQLite file path when type is sqlite.",
    )
    url: str | None = Field(
        default=None,
        description="Database URL when type is postgres.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "StructuredStoreConfig":
        if self.type == "sqlite" and not self.path:
            raise ValueError("structured_store.path is required for sqlite type")
        if self.type == "postgres" and not self.url:
            raise ValueError("structured_store.url is required for postgres type")
        return self


class VectorStoreConfig(ConfigPromptMixin, BaseModel):
    type: Literal["faiss", "pgvector"] = Field(
        default="faiss",
        description="Vector store backend."
    )
    path: str | None = Field(
        default=None,
        description="FAISS index path when type is faiss.",
    )
    url: str | None = Field(
        default=None,
        description="Database URL when type is pgvector.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "VectorStoreConfig":
        if self.type == "pgvector" and not self.url:
            raise ValueError("vector_store.url is required for pgvector type")
        return self


class DocumentStoreConfig(ConfigPromptMixin, BaseModel):
    type: Literal["local", "s3"] = Field(
        default="local",
        description="Document store backend."
    )
    path: str | None = Field(
        default=None,
        description="Local filesystem path when type is local.",
    )
    bucket: str | None = Field(
        default=None,
        description="S3 bucket name when type is s3.",
    )
    prefix: str = Field(
        default="",
        description="Optional S3 prefix when type is s3.",
    )
    region: str | None = Field(
        default=None,
        description="Optional AWS region when type is s3.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "DocumentStoreConfig":
        if self.type == "local" and not self.path:
            raise ValueError("document_store.path is required for local type")
        if self.type == "s3" and not self.bucket:
            raise ValueError("document_store.bucket is required for s3 type")
        return self


class LogStoreConfig(ConfigPromptMixin, BaseModel):
    type: Literal["local", "s3"] = Field(
        default="local",
        description="Append-only log store backend."
    )
    path: str | None = Field(
        default=None,
        description="Local filesystem root directory when type is local.",
    )
    bucket: str | None = Field(
        default=None,
        description="S3 directory bucket name when type is s3.",
    )
    prefix: str = Field(
        default="",
        description="Optional S3 key prefix when type is s3.",
    )
    region: str | None = Field(
        default=None,
        description="Optional AWS region when type is s3.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "LogStoreConfig":
        if self.type == "local" and not self.path:
            raise ValueError("log_store.path is required for local type")
        if self.type == "s3" and not self.bucket:
            raise ValueError("log_store.bucket is required for s3 type")
        return self
