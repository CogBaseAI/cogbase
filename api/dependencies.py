"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from api.app_cache import AppCache
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry


def get_system_store(request: Request) -> SystemStore:
    return request.app.state.system_store  # type: ignore[no-any-return]


def get_app_cache(request: Request) -> AppCache:
    return request.app.state.app_cache  # type: ignore[no-any-return]


def get_system_structured_store(request: Request) -> Any:
    return request.app.state.system_structured_store


def get_system_vector_store(request: Request) -> Any:
    return request.app.state.system_vector_store


def get_system_document_store(request: Request) -> Any:
    return request.app.state.system_document_store


def get_skill_registry(request: Request) -> SkillRegistry:
    return request.app.state.skill_registry  # type: ignore[no-any-return]


SystemStoreDep = Annotated[SystemStore, Depends(get_system_store)]
AppCacheDep = Annotated[AppCache, Depends(get_app_cache)]
SystemStructuredStoreDep = Annotated[Any, Depends(get_system_structured_store)]
SystemVectorStoreDep = Annotated[Any, Depends(get_system_vector_store)]
SystemDocumentStoreDep = Annotated[Any, Depends(get_system_document_store)]
SkillRegistryDep = Annotated[SkillRegistry, Depends(get_skill_registry)]
