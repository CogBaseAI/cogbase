"""CogBase FastAPI application entry point."""

from __future__ import annotations

import logging
import pathlib
import sys
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cogbase.config.config import AppConfig
from api.factory import build_app
from cogbase.embeddings import build_embedding
from cogbase.llms import build_llm
from cogbase.stores import build_document_store, build_structured_store, build_vector_store
from api.app_cache import AppCache
from api.routers.applications import router as applications_router
from api.routers.app_generate import router as generate_router
from api.routers.skills import router as skills_router
from api.routers.system import router as system_router
from api.system_config import SystemConfig
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry

format = '%(asctime)s [%(levelname)s] %(process)d %(threadName)s ' \
         '%(filename)s:%(lineno)d - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)

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

    system_resources = SystemResources()

    if system_cfg.structured_store is not None:
        system_resources.structured_store = build_structured_store(system_cfg.structured_store)
        logger.info("system structured_store type=%s", system_cfg.structured_store.type)

    if system_cfg.vector_store is not None:
        system_resources.vector_store = build_vector_store(system_cfg.vector_store)
        logger.info("system vector_store type=%s", system_cfg.vector_store.type)

    if system_cfg.document_store is not None:
        system_resources.document_store = build_document_store(system_cfg.document_store)
        logger.info("system document_store type=%s", system_cfg.document_store.type)

    if system_cfg.llm is not None:
        try:
            system_resources.llm = build_llm(system_cfg.llm)
            system_resources.llm_config = system_cfg.llm
            logger.info("system llm provider=%s model=%s", system_cfg.llm.provider, system_cfg.llm.model)
        except Exception as exc:
            logger.warning("system llm not initialized (configure via Settings): %s", exc)

    if system_cfg.embedding is not None:
        try:
            system_resources.embedder = build_embedding(system_cfg.embedding)
            system_resources.embedding_config = system_cfg.embedding
            logger.info("system embedding provider=%s model=%s", system_cfg.embedding.provider, system_cfg.embedding.model)
        except Exception as exc:
            logger.warning("system embedding not initialized (configure via Settings): %s", exc)

    # Apply runtime overrides persisted via PATCH /system/config — these win over YAML.
    from cogbase.config.models import EmbeddingConfig, LLMConfig
    overrides = await system_store.load_system_config_overrides()
    if "llm" in overrides:
        try:
            llm_cfg = LLMConfig.model_validate_json(overrides["llm"])
            system_resources.llm = build_llm(llm_cfg)
            system_resources.llm_config = llm_cfg
            logger.info("system llm restored from db provider=%s model=%s", llm_cfg.provider, llm_cfg.model)
        except Exception as exc:
            logger.warning("failed to restore llm override from db: %s", exc)
    if "embedding" in overrides:
        try:
            emb_cfg = EmbeddingConfig.model_validate_json(overrides["embedding"])
            system_resources.embedder = build_embedding(emb_cfg)
            system_resources.embedding_config = emb_cfg
            logger.info("system embedding restored from db provider=%s model=%s", emb_cfg.provider, emb_cfg.model)
        except Exception as exc:
            logger.warning("failed to restore embedding override from db: %s", exc)

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
            instance = await build_app(config, system=system_resources, app_status=record.status, task_store=system_store)
            app_cache.add(record.name, instance)
            logger.info("restored app name=%s", record.name)
        except Exception as exc:
            logger.warning("failed to restore app name=%s: %s", record.name, exc)

    app.state.system_store = system_store
    app.state.system_resources = system_resources
    app.state.skill_registry = skill_registry
    app.state.app_cache = app_cache

    yield

    await _close_store(system_db_store)
    if system_resources.structured_store is not None:
        await _close_store(system_resources.structured_store)
    if system_resources.vector_store is not None:
        await _close_store(system_resources.vector_store)
    if system_resources.document_store is not None:
        await _close_store(system_resources.document_store)


def _get_version() -> str:
    try:
        return version("cogbase")
    except PackageNotFoundError:
        return "latest"


app = FastAPI(
    title="CogBase API",
    description=(
        "Manage CogBase applications via REST. "
        "Each application is backed by an LLM provider, embedding model, "
        "structured store, and optional vector store, all configured via YAML."
    ),
    version=_get_version(),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(applications_router)
app.include_router(generate_router)
app.include_router(skills_router)
app.include_router(system_router)


# For production services, common pattern is nginx in front:
# nginx serves ui/dist/ directly as static files and reverse-proxies,
# /api/ (or similar prefix) to the Python process.
@app.get("/examples/demos", include_in_schema=False)
async def demo_catalog() -> dict:
    from examples.gen_demos_json import build_catalog
    return build_catalog()


_UI_DIST = pathlib.Path(__file__).parent.parent / "ui" / "dist"

if _UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_UI_DIST, html=True), name="ui")
