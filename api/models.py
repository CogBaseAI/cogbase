"""Request and response models for the CogBase REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ApplicationResponse(BaseModel):
    app_id: str
    name: str
    status: str   # "initializing" | "active" | "error"
    config: dict[str, Any]
    error: str | None
    created_at: str
    updated_at: str


class ApplicationListResponse(BaseModel):
    applications: list[ApplicationResponse]
    total: int
