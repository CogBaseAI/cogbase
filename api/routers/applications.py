"""CRUD endpoints for managing CogBase applications."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import json

import yaml

logger = logging.getLogger(__name__)
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.config import AppConfig
from api.dependencies import AppCacheDep, SystemConfigDep, SystemStoreDep, SystemStructuredStoreDep
from api.factory import build_app
from api.app_cache import AppCache
from api.models import (
    ApplicationListResponse,
    ApplicationResponse,
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    IngestResultResponse,
    QueryRequest,
    QueryResponse,
)
from api.system_config import SystemConfig
from api.system_store import AppRecord, SystemStore
from cogbase.core.models import Document

router = APIRouter(prefix="/applications", tags=["applications"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_response(record: AppRecord) -> ApplicationResponse:
    try:
        config_dict = yaml.safe_load(record.config_yaml) or {}
    except Exception:
        config_dict = {}
    return ApplicationResponse(
        name=record.name,
        status=record.status,
        config=config_dict,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _parse_config(raw: bytes) -> tuple[str, AppConfig]:
    yaml_text = raw.decode()
    try:
        config = AppConfig.from_yaml(yaml_text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config YAML: {exc}") from exc
    return yaml_text, config


@router.post("", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
async def create_application(
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    system_config: SystemConfigDep,
    system_structured_store: SystemStructuredStoreDep,
    config_file: UploadFile = File(..., description="YAML config file"),
) -> ApplicationResponse:
    """Create a new CogBase application from a YAML config file.

    The YAML must contain at minimum ``name`` and ``llm`` sections.  Store
    backends (``structured_store``, ``vector_store``) are optional — when
    omitted the service automatically uses the system-configured stores defined
    in ``cogbase_system.yaml``.

    The application is set up immediately; its status is ``active`` on success
    or ``error`` if setup fails (the record is still persisted so you can
    inspect the error and update the config).
    """
    yaml_text, config = _parse_config(await config_file.read())

    if await system_store.get_app(config.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application '{config.name}' already exists",
        )

    now = _now()
    record = AppRecord(
        name=config.name,
        config_yaml=yaml_text,
        status="initializing",
        created_at=now,
        updated_at=now,
    )
    await system_store.save_app(record)
    logger.info("Creating application '%s'", config.name)

    try:
        app = build_app(
            config,
            system_structured_store=system_structured_store,
            system_vector_store_cfg=system_config.vector_store,
        )
        await app.setup()
        app_cache.add(config.name, app)
        record = record.model_copy(update={"status": "active", "updated_at": _now()})
        logger.info("Application '%s' created successfully", config.name)
    except Exception as exc:
        logger.exception("Failed to create application '%s'", config.name)
        record = record.model_copy(
            update={"status": "error", "error": str(exc), "updated_at": _now()}
        )

    await system_store.save_app(record)
    return _to_response(record)


@router.get("", response_model=ApplicationListResponse)
async def list_applications(system_store: SystemStoreDep) -> ApplicationListResponse:
    """Return all registered applications."""
    records = await system_store.list_apps()
    items = [_to_response(r) for r in records]
    return ApplicationListResponse(applications=items, total=len(items))


@router.get("/{app_name}", response_model=ApplicationResponse)
async def get_application(
    app_name: str,
    system_store: SystemStoreDep,
) -> ApplicationResponse:
    """Return metadata for a single application."""
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    return _to_response(record)


@router.patch("/{app_name}", response_model=ApplicationResponse)
async def update_application(
    app_name: str,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    system_config: SystemConfigDep,
    system_structured_store: SystemStructuredStoreDep,
    config_file: UploadFile = File(..., description="Updated YAML config file"),
) -> ApplicationResponse:
    """Replace an application's config and restart it.

    The old instance is torn down before the new config is applied.  If the new
    config fails to initialise the application, the record is kept with
    ``status=error`` so you can inspect and fix the config.
    """
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    yaml_text, config = _parse_config(await config_file.read())

    if config.name != app_name and await system_store.get_app(config.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application '{config.name}' already exists",
        )

    app_cache.remove(app_name)

    if config.name != app_name:
        await system_store.delete_app(app_name)

    updated = record.model_copy(
        update={
            "name": config.name,
            "config_yaml": yaml_text,
            "status": "initializing",
            "error": None,
            "updated_at": _now(),
        }
    )
    await system_store.save_app(updated)
    logger.info("Updating application '%s'", app_name)

    try:
        app = build_app(
            config,
            system_structured_store=system_structured_store,
            system_vector_store_cfg=system_config.vector_store,
        )
        await app.setup()
        app_cache.add(config.name, app)
        updated = updated.model_copy(update={"status": "active", "updated_at": _now()})
        logger.info("Application '%s' updated successfully", config.name)
    except Exception as exc:
        logger.exception("Failed to update application '%s'", app_name)
        updated = updated.model_copy(
            update={"status": "error", "error": str(exc), "updated_at": _now()}
        )

    await system_store.save_app(updated)
    return _to_response(updated)


@router.delete("/{app_name}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_application(
    app_name: str,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
) -> None:
    """Permanently remove an application and its metadata."""
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    app_cache.remove(app_name)
    await system_store.delete_app(app_name)
    logger.info("Application '%s' deleted", app_name)


async def _get_active_app(
    app_name: str,
    app_cache: AppCache,
    system_store: SystemStore,
    system_config: SystemConfig,
    system_structured_store: object,
    *,
    force_refresh: bool = False,
) -> object:
    if not force_refresh:
        app = app_cache.get(app_name)
        if app is not None:
            return app
    else:
        app_cache.remove(app_name)
    record = await system_store.get_app(app_name)
    if record is None or record.status != "active":
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found or not active")
    config = AppConfig.from_yaml(record.config_yaml)
    app = build_app(
        config,
        system_structured_store=system_structured_store,
        system_vector_store_cfg=system_config.vector_store,
    )
    await app.setup()
    app_cache.add(app_name, app)
    return app


@router.post("/{app_name}/ingest_documents", response_model=IngestDocumentsResponse)
async def ingest_documents(
    app_name: str,
    body: IngestDocumentsRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_config: SystemConfigDep,
    system_structured_store: SystemStructuredStoreDep,
) -> IngestDocumentsResponse:
    """Ingest a batch of documents into an active application.

    Documents are processed concurrently up to *concurrency* at a time.  A
    failure on one document does not abort the others — each result carries
    ``success`` and ``error`` for per-document reporting.
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_config, system_structured_store)
    documents = [Document(doc_id=d.doc_id, text=d.text, metadata=d.metadata) for d in body.documents]
    try:
        results = await app.ingest_documents(documents, concurrency=body.concurrency)
    except Exception:
        logger.exception("ingest_documents failed for app '%s', retrying with fresh app", app_name)
        app = await _get_active_app(
            app_name, app_cache, system_store, system_config, system_structured_store, force_refresh=True
        )
        results = await app.ingest_documents(documents, concurrency=body.concurrency)
    return IngestDocumentsResponse(
        results=[
            IngestResultResponse(
                doc_id=r.doc_id,
                success=r.success,
                records_extracted=r.records_extracted,
                error=str(r.error) if r.error is not None else None,
            )
            for r in results
        ]
    )


async def _drain_query(app, text: str):
    """Drain app.query_stream and return the final GenerationResult."""
    async for item in app.query_stream(text):
        if not isinstance(item, str):
            return item
    raise RuntimeError("query_stream did not yield a GenerationResult")


@router.post("/{app_name}/query", response_model=QueryResponse)
async def query_application(
    app_name: str,
    body: QueryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_config: SystemConfigDep,
    system_structured_store: SystemStructuredStoreDep,
) -> QueryResponse:
    """Answer a natural-language query over an active application's ingested documents.

    The query is automatically routed to the appropriate retrieval pattern
    (A — structured lookup, B — semantic search, C — hybrid, D — grounded report).
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_config, system_structured_store)
    try:
        result = await _drain_query(app, body.text)
    except Exception:
        logger.exception("query failed for app '%s', retrying with fresh app", app_name)
        app = await _get_active_app(
            app_name, app_cache, system_store, system_config, system_structured_store, force_refresh=True
        )
        result = await _drain_query(app, body.text)
    return QueryResponse(
        answer=result.answer,
        pattern=result.pattern.value,
        findings=result.findings,
        supporting_quotes=result.supporting_quotes,
    )


@router.post("/{app_name}/query/stream")
async def query_application_stream(
    app_name: str,
    body: QueryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_config: SystemConfigDep,
    system_structured_store: SystemStructuredStoreDep,
) -> StreamingResponse:
    """Stream a natural-language query response as Server-Sent Events.

    Token events: ``{"token": "<text>"}``
    Final event:  ``{"result": {answer, pattern, findings, supporting_quotes}}``
    Sentinel:     ``data: [DONE]``
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_config, system_structured_store)

    async def event_stream():
        try:
            async for item in app.query_stream(body.text):
                if isinstance(item, str):
                    yield f"data: {json.dumps({'token': item})}\n\n"
                else:
                    payload = {
                        "result": {
                            "answer": item.answer,
                            "pattern": item.pattern.value,
                            "findings": item.findings,
                            "supporting_quotes": item.supporting_quotes,
                        }
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception:
            logger.exception("query_stream failed for app '%s'", app_name)
            yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
