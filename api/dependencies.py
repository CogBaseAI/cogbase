"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from api.app_cache import AppCache
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.store import SkillBundleStore


def get_system_store(request: Request) -> SystemStore:
    return request.app.state.system_store  # type: ignore[no-any-return]


def get_app_cache(request: Request) -> AppCache:
    return request.app.state.app_cache  # type: ignore[no-any-return]


def get_system_resources(request: Request) -> SystemResources:
    return request.app.state.system_resources  # type: ignore[no-any-return]


def get_skill_registry(request: Request) -> SkillRegistry:
    return request.app.state.skill_registry  # type: ignore[no-any-return]


def get_skill_bundle_store(request: Request) -> SkillBundleStore:
    store = request.app.state.skill_bundle_store
    if store is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Skill uploads require a system document store; none is configured.",
        )
    return store  # type: ignore[no-any-return]


SystemStoreDep = Annotated[SystemStore, Depends(get_system_store)]
AppCacheDep = Annotated[AppCache, Depends(get_app_cache)]
SystemResourcesDep = Annotated[SystemResources, Depends(get_system_resources)]
SkillRegistryDep = Annotated[SkillRegistry, Depends(get_skill_registry)]
SkillBundleStoreDep = Annotated[SkillBundleStore, Depends(get_skill_bundle_store)]
