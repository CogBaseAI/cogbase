"""CogBase FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from cogbase.config.config import AppConfig
from api.factory import build_app, build_structured_store
from api.app_cache import AppCache
from api.routers.applications import router as applications_router
from api.routers.skills import router as skills_router
from api.system_config import SystemConfig
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


async def _close_store(store: object) -> None:
    """Close a store that may have a sync or async ``close`` method."""
    closer = getattr(store, "close", None)
    if closer is None:
        return
    result = closer()
    if result is not None:
        import inspect
        if inspect.isawaitable(result):
            await result


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load system config from file / env vars / defaults.
    system_cfg = SystemConfig.load()
    logger.info("system_config loaded system_db=%s", system_cfg.system_db)

    system_db_store = build_structured_store(system_cfg.system_db)
    system_store = SystemStore(store=system_db_store)
    await system_store.setup()

    # Build the shared structured store (None when not configured).
    system_structured_store = None
    if system_cfg.structured_store is not None:
        system_structured_store = build_structured_store(system_cfg.structured_store)
        logger.info(
            "system_structured_store type=%s", system_cfg.structured_store.type
        )

    skill_registry = SkillRegistry()
    if system_cfg.skills_dir is not None:
        skill_registry.load_from_dir(system_cfg.skills_dir)
        logger.info("skill_registry loaded from skills_dir=%s", system_cfg.skills_dir)

    app_cache = AppCache()

    # Re-instantiate all previously active applications so they are immediately
    # usable without a POST /applications round-trip.
    for record in await system_store.list_apps():
        if record.status != "active":
            continue
        try:
            config = AppConfig.from_yaml(record.config_yaml)
            instance = build_app(
                config,
                system_structured_store=system_structured_store,
                system_vector_store_cfg=system_cfg.vector_store,
                system_document_store_cfg=system_cfg.document_store,
            )
            await instance.setup()
            app_cache.add(record.name, instance)
            logger.info("restored app name=%s", record.name)
        except Exception as exc:
            logger.warning("failed to restore app name=%s: %s", record.name, exc)

    app.state.system_config = system_cfg
    app.state.system_structured_store = system_structured_store
    app.state.system_store = system_store
    app.state.skill_registry = skill_registry
    app.state.app_cache = app_cache

    yield

    await _close_store(system_db_store)
    if system_structured_store is not None:
        await _close_store(system_structured_store)


app = FastAPI(
    title="CogBase API",
    description=(
        "Manage CogBase applications via REST. "
        "Each application is backed by an LLM provider, embedding model, "
        "structured store, and optional vector store, all configured via YAML."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(applications_router)
app.include_router(skills_router)
