"""Store configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class StructuredStoreConfig(BaseModel):
    type: Literal["sqlite", "postgres", "memory"] = "memory"
    path: str | None = None  # sqlite only
    url: str | None = None   # postgres only

    @model_validator(mode="after")
    def _validate(self) -> "StructuredStoreConfig":
        if self.type == "sqlite" and not self.path:
            raise ValueError("structured_store.path is required for sqlite type")
        if self.type == "postgres" and not self.url:
            raise ValueError("structured_store.url is required for postgres type")
        return self


class VectorStoreConfig(BaseModel):
    type: Literal["faiss", "pgvector"] = "faiss"
    url: str | None = None  # pgvector only

    @model_validator(mode="after")
    def _validate(self) -> "VectorStoreConfig":
        if self.type == "pgvector" and not self.url:
            raise ValueError("vector_store.url is required for pgvector type")
        return self


class DocumentStoreConfig(BaseModel):
    type: Literal["local", "s3"] = "local"
    path: str | None = None      # local only
    bucket: str | None = None    # s3 only
    prefix: str = ""             # s3 only
    region: str | None = None    # s3 only

    @model_validator(mode="after")
    def _validate(self) -> "DocumentStoreConfig":
        if self.type == "local" and not self.path:
            raise ValueError("document_store.path is required for local type")
        if self.type == "s3" and not self.bucket:
            raise ValueError("document_store.bucket is required for s3 type")
        return self
