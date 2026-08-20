"""CogBase FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import AsyncIterator, Callable

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cogbase.config.config import AppConfig
from api.auth import require_configured_jwt_secret, set_jwt_secret
from api.dependencies import (
    set_deployment_mode,
    set_system_config_writable,
    set_tenant_skill_upload,
)
from api.factory import build_app
from cogbase.embeddings import build_embedding
from cogbase.llms import build_llm
from cogbase.stores import (
    build_document_store,
    build_log_store,
    build_structured_store,
    build_vector_store,
)
from api.app_cache import AppCache, cache_key
from api.routers.applications import router as applications_router
from api.routers.applications import account_router as applications_account_router
from api.routers.auth import router as auth_router
from api.routers.app_generate import router as generate_router
from api.routers.app_generate import deploy_router as generate_deploy_router
from api.routers.namespaces import router as namespaces_router
from api.routers.profile import router as profile_router
from api.routers.skills import router as skills_router
from api.routers.system import router as system_router
from api.routers.whoami import router as whoami_router
from api.system_config import SystemConfig
from api.system_resources import SystemResources
from api.system_store import SystemStore
from api.task_runner import recover_orphaned_tasks
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.skill import load_skill_dir
from cogbase.skills.store import SkillBundleStore

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


_SECRET_ENV_VARS = (
    "COGBASE_JWT_SECRET",
    "COGBASE_LLM_API_KEY",
    "COGBASE_EMBEDDING_API_KEY",
    "COGBASE_POSTGRES_DSN",
    "COGBASE_PGVECTOR_DSN",
)


def _apply_secret_env_overrides(system_cfg: SystemConfig, env: dict[str, str]) -> None:
    """Secrets-never-on-disk: let jwt_secret, the provider api_keys, and the
    Postgres DSN(s) arrive as env vars instead of (or on top of) whatever the
    system YAML has, so an image's YAML need carry no real secret.

    Pure w.r.t. ``os.environ`` — takes an explicit mapping — so it is testable
    without exercising a real store connection. ``lifespan`` calls this with
    ``os.environ`` itself, before anything reads ``system_cfg.jwt_secret``/
    ``llm``/``embedding``/store urls, and deletes the same vars from
    ``os.environ`` before ``yield`` — see the callers for why.
    """
    if secret := env.get("COGBASE_JWT_SECRET"):
        system_cfg.jwt_secret = secret
    if system_cfg.llm and (key := env.get("COGBASE_LLM_API_KEY")):
        system_cfg.llm.api_key = key
    if system_cfg.embedding and (key := env.get("COGBASE_EMBEDDING_API_KEY")):
        system_cfg.embedding.api_key = key
    # A "postgres"/"pgvector" type in the YAML declares the backend, but the
    # credential-bearing url comes from the environment. One deployment
    # typically points system_db and structured_store at the same instance
    # (per docs/launch-plan.md §C); pgvector falls back to the same DSN when
    # no separate one is given, since pgvector is usually the same Postgres
    # instance with the extension enabled.
    if (pg_dsn := env.get("COGBASE_POSTGRES_DSN")):
        if system_cfg.system_db.type == "postgres":
            system_cfg.system_db.url = pg_dsn
        if system_cfg.structured_store and system_cfg.structured_store.type == "postgres":
            system_cfg.structured_store.url = pg_dsn
    if system_cfg.vector_store and system_cfg.vector_store.type == "pgvector":
        pgvector_dsn = env.get("COGBASE_PGVECTOR_DSN") or env.get("COGBASE_POSTGRES_DSN")
        if pgvector_dsn:
            system_cfg.vector_store.url = pgvector_dsn


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load system config from file / env vars / defaults.
    system_cfg = SystemConfig.load()
    # system_db.url carries the DSN — credentials and all, for a postgres
    # backend — so only its type is safe to put in a log line.
    logger.info("system_config loaded system_db_type=%s mode=%s",
                system_cfg.system_db.type, system_cfg.deployment_mode)

    # Applied before anything reads system_cfg.jwt_secret/llm/embedding/store
    # urls; the env vars themselves are deleted below, before `yield`, so no
    # later request-time code — including a tenant-triggered subprocess
    # inheriting os.environ — can read them back.
    _apply_secret_env_overrides(system_cfg, os.environ)

    # Apply the operator-declared mode and signing secret, then refuse to boot a
    # managed deployment on the forgeable dev secret — in saas mode the access
    # token is the tenant boundary.
    set_deployment_mode(system_cfg.deployment_mode)
    set_jwt_secret(system_cfg.jwt_secret)
    require_configured_jwt_secret(system_cfg.deployment_mode)
    set_tenant_skill_upload(system_cfg.tenant_skill_upload)
    set_system_config_writable(system_cfg.system_config_writable)

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

    if system_cfg.log_store is not None:
        system_resources.log_store = build_log_store(system_cfg.log_store)
        logger.info("system log_store type=%s", system_cfg.log_store.type)

    if system_cfg.llm is not None:
        try:
            system_resources.llm = build_llm(system_cfg.llm)
            system_resources.llm_config = system_cfg.llm
            logger.info("system llm provider=%s model=%s mini_model=%s",
                        system_cfg.llm.provider, system_cfg.llm.model, system_cfg.llm.mini_model)
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

    # Uploaded skills live in the system document store (the shared, multi-node
    # source of truth). Materialize each into the local cache and register it so
    # this node — even a freshly started one — sees skills uploaded elsewhere.
    skill_bundle_store: SkillBundleStore | None = None
    if system_resources.document_store is not None:
        skill_bundle_store = SkillBundleStore(system_resources.document_store)
        for sk in await system_store.list_skills():
            try:
                skill_root = await skill_bundle_store.sync_from_store(sk.skill_id)
                skill = load_skill_dir(skill_root, skill_id=sk.skill_id)
                if skill is not None:
                    skill_registry.register(skill, account_id=sk.account_id, replace=True)
                    logger.info(
                        "synced skill id=%s name=%s account=%s from document store",
                        sk.skill_id, sk.name, sk.account_id,
                    )
            except Exception as exc:
                logger.warning("failed to sync skill id=%s: %s", sk.skill_id, exc)

    # Make the registry available to build_app (skills are a system-level resource).
    system_resources.skill_registry = skill_registry

    app_cache = AppCache()

    # Re-instantiate all previously active applications so they are immediately
    # usable without a POST /applications round-trip.
    for record in await system_store.list_apps():
        if record.status != "active":
            continue
        try:
            config = AppConfig.from_yaml(record.config_yaml)
            instance = await build_app(
                config, app_id=record.app_id, account_id=record.account_id,
                namespace_id=record.namespace_id, system=system_resources,
                app_status=record.status, task_store=system_store,
            )
            app_cache.add(cache_key(record.account_id, record.namespace_id, record.name), instance)
            logger.info(
                "restored app name=%s account=%s namespace=%s",
                record.name, record.account_id, record.namespace_id,
            )
        except Exception as exc:
            logger.warning("failed to restore app name=%s: %s", record.name, exc)

    app.state.system_store = system_store
    app.state.system_resources = system_resources
    app.state.skill_registry = skill_registry
    app.state.skill_bundle_store = skill_bundle_store
    app.state.app_cache = app_cache
    # Set so the attribute always exists and is discoverable here; a deployment
    # serving more than one vertical replaces it after the lifespan has run (or
    # reuses this lifespan from its own app and sets it there). ``None`` means
    # every account gets COGBASE_INTERVIEW_SKILL — see
    # cogbase/core/onboarding.py::InterviewSkillResolver.
    app.state.interview_skill_resolver = None

    # Requeue background tasks (ingest/distill/workflow) left unfinished by a
    # previous process: an in-process create_task is lost on crash/deploy/OOM,
    # stranding its task record in PENDING/RUNNING. Re-execution is idempotent.
    # Run in the background so startup stays fast.
    async def _resolve_app(rec) -> object | None:
        # ``rec`` is the AppRecord (carries account/namespace); resolve by its
        # scoped cache key, rebuilding from the config if not warm.
        key = cache_key(rec.account_id, rec.namespace_id, rec.name)
        inst = app_cache.get(key)
        if inst is not None:
            return inst
        if rec.status != "active":
            return None
        built = await build_app(
            AppConfig.from_yaml(rec.config_yaml), app_id=rec.app_id,
            account_id=rec.account_id, namespace_id=rec.namespace_id,
            system=system_resources, app_status=rec.status, task_store=system_store,
        )
        app_cache.add(key, built)
        return built

    async def _recover() -> None:
        try:
            await recover_orphaned_tasks(system_store, _resolve_app, app_cache)
        except Exception:
            logger.exception("startup task recovery failed")

    asyncio.create_task(_recover())

    # Everything above that needed these has already read them (set_jwt_secret,
    # build_llm/build_embedding baking the key into their HTTP clients). Nothing
    # after this point may reconstruct a secret from os.environ — most notably
    # cogbase.core.query_runner._tool_env, which hands a tenant-triggered
    # subprocess a copy of the process environment.
    for var in _SECRET_ENV_VARS:
        os.environ.pop(var, None)

    yield

    await _close_store(system_db_store)
    if system_resources.structured_store is not None:
        await _close_store(system_resources.structured_store)
    if system_resources.vector_store is not None:
        await _close_store(system_resources.vector_store)
    if system_resources.document_store is not None:
        await _close_store(system_resources.document_store)
    if system_resources.log_store is not None:
        await _close_store(system_resources.log_store)


def _get_version() -> str:
    try:
        return version("cogbase")
    except PackageNotFoundError:
        return "latest"


Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]

_UI_DIST = pathlib.Path(__file__).parent.parent / "ui" / "dist"


def create_app(
    *,
    extra_routers: Sequence[APIRouter] = (),
    extra_lifespan: Lifespan | None = None,
    mount_ui: bool = True,
) -> FastAPI:
    """Build the CogBase FastAPI app.

    A module-scoped ``app = FastAPI(...)`` cannot be composed by an embedder:
    ``ui/dist`` is mounted at ``/`` as the last route, a Starlette ``Mount("/")``
    matches every path, and ``app.include_router(...)`` called afterwards from
    outside this module is dead code. Everything that changes the app — routers,
    lifespan — therefore has to happen inside this function, before the UI mount
    at the end, which is what ``extra_routers`` and ``extra_lifespan`` are for.

    ``extra_lifespan`` *replaces* :func:`lifespan` rather than composing with it
    automatically: the two things an embedder plausibly wants to do — run setup
    before this module's lifespan touches ``skills_dir``, and run setup after it
    has built ``app.state.system_store`` — sit on opposite sides of this
    lifespan's body, so no single composition order serves both. An embedder
    that wants either (or both) writes its own lifespan that wraps
    :func:`lifespan` itself and passes that as ``extra_lifespan``.

    ``mount_ui=False`` skips mounting this repo's own ``ui/dist``, for the same
    reason ``extra_routers`` exists: the mount is a Starlette ``Mount("/")``, so
    an embedder that wants to serve its own UI at ``/`` instead has to stop this
    function from claiming that route first, then mount its own build onto the
    app this function returns.
    """
    app = FastAPI(
        title="CogBase API",
        description=(
            "Manage CogBase applications via REST. "
            "Each application is backed by an LLM provider, embedding model, "
            "structured store, and optional vector store, all configured via YAML."
        ),
        version=_get_version(),
        lifespan=extra_lifespan if extra_lifespan is not None else lifespan,
    )

    # Origins allowed to call the API from a browser. Defaults to "*" for local
    # dev; a real deployment sets COGBASE_ALLOWED_ORIGINS to a comma-separated
    # allowlist (e.g. "https://app.example.com"). Auth uses Bearer tokens, not
    # cookies, so a wildcard origin carries no credential-leak risk
    # (allow_credentials stays off).
    origins_env = os.environ.get("COGBASE_ALLOWED_ORIGINS", "*").strip()
    allow_origins = (
        ["*"] if origins_env == "*"
        else [o.strip() for o in origins_env.split(",") if o.strip()]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (
        namespaces_router,
        applications_router,
        applications_account_router,
        auth_router,
        generate_router,
        generate_deploy_router,
        profile_router,
        skills_router,
        system_router,
        whoami_router,
        *extra_routers,
    ):
        app.include_router(router)

    @app.get("/health", include_in_schema=False)
    async def health() -> dict:
        """Liveness + readiness probe.

        Reports ``ok`` once the system store is reachable; deployment health
        checks and load balancers poll this. Kept dependency-light so it stays
        fast.
        """
        try:
            store = app.state.system_store
            await store.list_namespaces("__healthcheck__")  # cheap scoped read
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("health check failed: %s", exc)
            raise HTTPException(status_code=503, detail="not ready")

    # For production services, common pattern is nginx in front:
    # nginx serves ui/dist/ directly as static files and reverse-proxies,
    # /api/ (or similar prefix) to the Python process.
    @app.get("/examples/demos", include_in_schema=False)
    async def demo_catalog() -> dict:
        from examples.gen_demos_json import build_catalog
        return build_catalog()

    if mount_ui and _UI_DIST.is_dir():
        app.mount("/", StaticFiles(directory=_UI_DIST, html=True), name="ui")

    return app


app = create_app()
