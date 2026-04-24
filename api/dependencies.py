"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from api.app_cache import AppCache
from api.system_config import SystemConfig
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry


def get_system_store(request: Request) -> SystemStore:
    return request.app.state.system_store  # type: ignore[no-any-return]


def get_app_cache(request: Request) -> AppCache:
    return request.app.state.app_cache  # type: ignore[no-any-return]


def get_system_config(request: Request) -> SystemConfig:
    return request.app.state.system_config  # type: ignore[no-any-return]


def get_system_structured_store(request: Request) -> Any:
    return request.app.state.system_structured_store


def get_skill_registry(request: Request) -> SkillRegistry:
    return request.app.state.skill_registry  # type: ignore[no-any-return]


SystemStoreDep = Annotated[SystemStore, Depends(get_system_store)]
AppCacheDep = Annotated[AppCache, Depends(get_app_cache)]
SystemConfigDep = Annotated[SystemConfig, Depends(get_system_config)]
SystemStructuredStoreDep = Annotated[Any, Depends(get_system_structured_store)]
SkillRegistryDep = Annotated[SkillRegistry, Depends(get_skill_registry)]
