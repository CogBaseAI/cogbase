"""CogBase FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.config import AppConfig
from api.factory import build_app
from api.registry import AppRegistry
from api.routers.applications import router as applications_router
from api.system_store import SystemStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db_path = os.environ.get("COGBASE_SYSTEM_DB", "./cogbase_system.db")
    system_store = SystemStore(db_path=db_path)
    await system_store.setup()

    registry = AppRegistry()

    # Re-instantiate all previously active applications so they are immediately
    # usable without a POST /applications round-trip.
    for record in await system_store.list_apps():
        if record.status != "active":
            continue
        try:
            config = AppConfig.from_yaml(record.config_yaml)
            instance = build_app(config)
            await instance.setup()
            registry.add(record.app_id, instance)
            logger.info("restored app name=%s app_id=%s", record.name, record.app_id)
        except Exception as exc:
            logger.warning(
                "failed to restore app name=%s app_id=%s: %s",
                record.name,
                record.app_id,
                exc,
            )

    app.state.system_store = system_store
    app.state.registry = registry

    yield

    system_store.close()


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
