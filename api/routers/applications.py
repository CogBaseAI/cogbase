"""CRUD endpoints for managing CogBase applications."""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
import zipfile
from datetime import datetime, timezone

import json
import yaml

logger = logging.getLogger(__name__)
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from cogbase.config.config import AppConfig
from api.dependencies import AppCacheDep, SkillRegistryDep, SystemResourcesDep, SystemStoreDep
from api.system_resources import SystemResources
from api.factory import build_app
from api.app_cache import AppCache
from api.models import (
    AddSkillRequest,
    AppSkillsResponse,
    ApplicationListResponse,
    ApplicationResponse,
    ChunkResponse,
    CollectionsResponse,
    DocListResponse,
    DocResponse,
    FilterRequest,
    IngestDocumentsAcceptedResponse,
    QueryRequest,
    QueryResponse,
    CollectionQueryRequest,
    CollectionQueryResponse,
    TaskListResponse,
    TaskResponse,
    WorkflowDocListResponse,
    WorkflowListResponse,
    WorkflowRunRequest,
)
from api.system_store import AppRecord, DocRecord, SystemStore, TaskRecord
from cogbase.core.models import Document
from cogbase.pipeline.document_parser import parse_to_markdown
from cogbase.stores.filters import Filter, Op

router = APIRouter(prefix="/applications", tags=["applications"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _doc_to_response(record: DocRecord) -> DocResponse:
    import json as _json
    try:
        meta = _json.loads(record.metadata) if record.metadata else {}
    except Exception:
        meta = {}
    return DocResponse(
        doc_id=record.doc_id,
        app_name=record.app_name,
        status=record.status,
        ingested_at=record.ingested_at,
        metadata=meta,
    )


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


def _resolve_step_refs(step: dict, files: dict[str, str]) -> None:
    """Resolve file references in a single workflow step (recurses into foreach steps)."""
    prompt_ref = step.get("prompt", "")
    if prompt_ref and prompt_ref in files:
        step["prompt"] = files[prompt_ref]
    schema_ref = step.get("output_schema", "")
    if schema_ref and schema_ref in files:
        step["output_schema"] = files[schema_ref]
    for substep in step.get("steps") or []:
        _resolve_step_refs(substep, files)


def _resolve_file_refs(data: dict, files: dict[str, str]) -> None:
    """Replace filename references in-place with file contents from the ZIP."""
    for sc in data.get("structured_collections", []):
        schema_ref = sc.get("schema", "")
        if schema_ref in files:
            sc["schema"] = files[schema_ref]

    for pipeline in data.get("pipelines", []):
        for step in pipeline.get("steps", []):
            extractor = step.get("extractor") or {}
            schema_ref = extractor.get("extraction_schema", "")
            if schema_ref and schema_ref in files:
                extractor["extraction_schema"] = files[schema_ref]
            prompt_ref = extractor.get("prompt", "")
            if prompt_ref and prompt_ref in files:
                extractor["prompt"] = files[prompt_ref]
            prompt_ref = step.get("doc_prompt", "")
            if prompt_ref and prompt_ref in files:
                step["doc_prompt"] = files[prompt_ref]

    for wf in data.get("workflows", []):
        for step in wf.get("steps", []):
            _resolve_step_refs(step, files)




def _parse_bundle(raw: bytes) -> tuple[str, AppConfig]:
    """Unzip bundle, resolve file refs, parse config, return (stored_yaml, config)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid ZIP archive") from exc

    names = set(zf.namelist())
    if "config.yaml" not in names:
        raise HTTPException(status_code=422, detail="ZIP bundle must contain config.yaml at the root")

    try:
        yaml_text = zf.read("config.yaml").decode()
        files = {n: zf.read(n).decode() for n in names if n != "config.yaml"}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to read bundle contents: {exc}") from exc

    try:
        data = yaml.safe_load(yaml_text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config.yaml: {exc}") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="config.yaml must be a YAML mapping")

    _resolve_file_refs(data, files)

    try:
        config = AppConfig.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config: {exc}") from exc

    # Store the resolved config (file refs replaced with content) so the app
    # can be rebuilt from the system store without the original ZIP.
    stored_yaml = config.to_yaml()
    return stored_yaml, config


def _to_filter(fr: FilterRequest) -> Filter:
    try:
        op = Op(fr.op)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown filter op: {fr.op!r}")
    return Filter(field=fr.field, op=op, value=fr.value)


def _validate_skills(skill_names: list[str], skill_registry) -> None:
    """Raise HTTP 422 if any skill name is not in the registry."""
    unknown = []
    for name in skill_names:
        try:
            skill_registry.get(name)
        except KeyError:
            unknown.append(name)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown skill(s): {', '.join(unknown)}. Run GET /skills to see available skills.",
        )


@router.post("", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
async def create_application(
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    system_resources: SystemResourcesDep,
    skill_registry: SkillRegistryDep,
    bundle: UploadFile = File(..., description="ZIP bundle containing config.yaml and referenced files"),
) -> ApplicationResponse:
    """Create a new CogBase application from a ZIP bundle.

    The bundle must contain ``config.yaml`` at the root.  Any files referenced
    by filename in the config (prompt templates, JSON schemas) must also be
    present flat at the zip root.

    The application is set up immediately; its status is ``active`` on success
    or ``error`` if setup fails (the record is still persisted so you can
    inspect the error and update the config).
    """
    yaml_text, config = _parse_bundle(await bundle.read())

    if config.skills:
        _validate_skills(config.skills, skill_registry)

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
        app = await build_app(config, system=system_resources, app_status=record.status)
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
    logger.info("list apps=%d", len(items))
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
    system_resources: SystemResourcesDep,
    skill_registry: SkillRegistryDep,
    bundle: UploadFile = File(..., description="Updated ZIP bundle containing config.yaml and referenced files"),
) -> ApplicationResponse:
    """Replace an application's config and restart it.

    The old instance is torn down before the new config is applied.  If the new
    config fails to initialise the application, the record is kept with
    ``status=error`` so you can inspect and fix the config.
    """
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    yaml_text, config = _parse_bundle(await bundle.read())

    if config.skills:
        _validate_skills(config.skills, skill_registry)

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
        app = await build_app(config, system=system_resources, app_status=updated.status)
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
    # TODO delete actual data from document/vector/structured stores.
    app_cache.remove(app_name)
    await system_store.delete_app(app_name)
    logger.info("Application '%s' deleted", app_name)


async def _get_active_app(
    app_name: str,
    app_cache: AppCache,
    system_store: SystemStore,
    system_resources: SystemResources,
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
    app = await build_app(config, system=system_resources, app_status=record.status, task_store=system_store)
    app_cache.add(app_name, app)
    return app


@router.post("/{app_name}/upload_documents", response_model=IngestDocumentsAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    app_name: str,
    files: list[UploadFile],
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
    # Optional JSON object applied to every file in this batch — use for pipeline
    # routing (match conditions) and workflow triggers (when.metadata).  For
    # per-file metadata, make separate upload calls.
    metadata: str = Form(default="{}"),
) -> IngestDocumentsAcceptedResponse:
    """Upload files and queue them for background ingestion.

    Files are parsed to markdown synchronously; ingestion runs in the background.
    Returns task IDs immediately (HTTP 202). Poll GET /{app_name}/tasks?task_type=ingest.
    """
    import json as _json_mod
    import pathlib
    import re

    try:
        extra_metadata: dict = _json_mod.loads(metadata)
    except _json_mod.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"metadata must be a valid JSON object: {exc}")
    if not isinstance(extra_metadata, dict):
        raise HTTPException(status_code=422, detail="metadata must be a JSON object")

    app = await _get_active_app(app_name, app_cache, system_store, system_resources)

    def _safe_doc_id(filename: str) -> str:
        stem = pathlib.Path(filename).stem
        return re.sub(r"[^\w\-]", "_", stem)

    now = _now()
    task_doc_pairs: list[tuple[str, Document]] = []
    original_saves: list[tuple[str, bytes, str]] = []
    all_task_ids: list[str] = []

    for upload in files:
        filename = upload.filename or "upload"
        suffix = pathlib.Path(filename).suffix.lower()
        doc_id = _safe_doc_id(filename)
        task_id = str(uuid.uuid4())
        all_task_ids.append(task_id)

        content = await upload.read()
        try:
            markdown_text = parse_to_markdown(content, filename)
        except Exception as exc:
            await system_store.create_task(TaskRecord(
                task_id=task_id,
                app_name=app_name,
                task_type="ingest",
                task_name="ingest",
                doc_id=doc_id,
                status="failed",
                started_at=now,
                completed_at=now,
                error=f"Failed to parse {filename!r}: {exc}",
            ))
            continue

        await system_store.create_task(TaskRecord(
            task_id=task_id,
            app_name=app_name,
            task_type="ingest",
            task_name="ingest",
            doc_id=doc_id,
            status="pending",
            started_at=now,
        ))
        original_saves.append((doc_id, content, suffix))
        task_doc_pairs.append((task_id, Document(
            doc_id=doc_id,
            text=markdown_text,
            metadata={"source_filename": filename, "source_format": suffix.lstrip("."), **extra_metadata},
        )))

    # Save original binary files to document store if configured.
    if app._document_store is not None:
        for doc_id, content, suffix in original_saves:
            try:
                await app._document_store.save_bytes(
                    app.name, f"originals/{doc_id}{suffix}", content
                )
            except Exception:
                logger.exception(
                    "upload_documents: failed to save original file doc_id=%s", doc_id
                )

    if task_doc_pairs:
        async def _run_upload_bg() -> None:
            async def _ingest_one(task_id: str, doc: Document) -> None:
                await system_store.update_task(task_id, status="running", started_at=_now())
                try:
                    current_app = app_cache.get(app_name) or app
                    results = await current_app.ingest_documents([doc], concurrency=1)
                    result = results[0]
                    if result.success:
                        await system_store.update_task(task_id, status="done", completed_at=_now())
                        await system_store.save_doc(DocRecord(
                            app_name=app_name,
                            doc_id=doc.doc_id,
                            status="active",
                            ingested_at=_now(),
                            metadata=json.dumps(doc.metadata) if doc.metadata else None,
                        ))
                    else:
                        await system_store.update_task(
                            task_id, status="failed", completed_at=_now(),
                            error=str(result.error) if result.error else "ingest failed",
                        )
                except Exception as exc:
                    logger.exception("upload_bg failed app=%s doc_id=%s", app_name, doc.doc_id)
                    await system_store.update_task(
                        task_id, status="failed", completed_at=_now(), error=str(exc)
                    )

            await asyncio.gather(*(_ingest_one(tid, doc) for tid, doc in task_doc_pairs))

        asyncio.create_task(_run_upload_bg())

    return IngestDocumentsAcceptedResponse(task_ids=all_task_ids, total=len(all_task_ids))


async def _drain_query(app, text: str, history: list[dict] | None = None):
    """Drain app.query_stream and return the final result."""
    async for item in app.query_stream(text, history=history):
        if not isinstance(item, str):
            return item
    raise RuntimeError("query_stream did not yield a result")


@router.post("/{app_name}/query", response_model=QueryResponse)
async def query_application(
    app_name: str,
    body: QueryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> QueryResponse:
    """Answer a natural-language query over an active application's ingested documents.

    The LLM agent loop calls structured_lookup and/or vector_search tools as needed,
    then synthesises a final answer.  Large structured result sets are returned
    directly (passthrough=True) without an additional synthesis step.
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    history = [{"role": m.role, "content": m.content} for m in body.history] or None
    try:
        result = await _drain_query(app, body.text, history=history)
    except Exception:
        logger.exception("query failed for app '%s', retrying with fresh app", app_name)
        app = await _get_active_app(
            app_name, app_cache, system_store, system_resources, force_refresh=True
        )
        result = await _drain_query(app, body.text, history=history)
    return QueryResponse(
        answer=result.answer,
        structured_records=result.structured_records,
        chunks=[ChunkResponse(**c.model_dump(exclude={"embedding"})) for c in result.chunks],
    )


@router.post("/{app_name}/query/stream")
async def query_application_stream(
    app_name: str,
    body: QueryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> StreamingResponse:
    """Stream a natural-language query response as Server-Sent Events.

    Token events: ``{"token": "<text>"}``
    Final event:  ``{"result": {answer, structured_records, chunks}}``
    Sentinel:     ``data: [DONE]``
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    history = [{"role": m.role, "content": m.content} for m in body.history] or None

    async def event_stream():
        try:
            async for item in app.query_stream(body.text, history=history):
                if isinstance(item, str):
                    yield f"data: {json.dumps({'token': item})}\n\n"
                else:
                    payload = {
                        "result": {
                            "answer": item.answer,
                            "structured_records": item.structured_records,
                            "chunks": [c.model_dump(exclude={"embedding"}) for c in item.chunks],
                        }
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception:
            logger.exception("query_stream failed for app '%s'", app_name)
            yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Application skills endpoints
# ---------------------------------------------------------------------------


@router.get("/{app_name}/skills", response_model=AppSkillsResponse)
async def list_application_skills(
    app_name: str,
    system_store: SystemStoreDep,
) -> AppSkillsResponse:
    """Return the skills currently assigned to an application."""
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    config = AppConfig.from_yaml(record.config_yaml)
    return AppSkillsResponse(app_name=app_name, skills=config.skills)


@router.post("/{app_name}/skills", response_model=AppSkillsResponse, status_code=status.HTTP_201_CREATED)
async def add_application_skill(
    app_name: str,
    body: AddSkillRequest,
    system_store: SystemStoreDep,
    skill_registry: SkillRegistryDep,
) -> AppSkillsResponse:
    """Assign a system skill to an application.

    The skill must exist in the system skill registry (configured via
    ``skills_dir`` in ``cogbase_system.yaml``).  Adding the same skill twice
    is idempotent.
    """
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    try:
        skill_registry.get(body.skill_name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{body.skill_name}' not found in the system skill registry",
        )

    config = AppConfig.from_yaml(record.config_yaml)
    if body.skill_name not in config.skills:
        updated_config = config.model_copy(update={"skills": config.skills + [body.skill_name]})
        updated_record = record.model_copy(
            update={"config_yaml": updated_config.to_yaml(), "updated_at": _now()}
        )
        await system_store.save_app(updated_record)
        logger.info("Added skill '%s' to application '%s'", body.skill_name, app_name)
        config = updated_config

    return AppSkillsResponse(app_name=app_name, skills=config.skills)


@router.delete("/{app_name}/skills/{skill_name}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def remove_application_skill(
    app_name: str,
    skill_name: str,
    system_store: SystemStoreDep,
) -> None:
    """Remove a skill from an application."""
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    config = AppConfig.from_yaml(record.config_yaml)
    if skill_name not in config.skills:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{skill_name}' is not assigned to application '{app_name}'",
        )

    updated_config = config.model_copy(update={"skills": [s for s in config.skills if s != skill_name]})
    updated_record = record.model_copy(
        update={"config_yaml": updated_config.to_yaml(), "updated_at": _now()}
    )
    await system_store.save_app(updated_record)
    logger.info("Removed skill '%s' from application '%s'", skill_name, app_name)


@router.get("/{app_name}/collections", response_model=CollectionsResponse)
async def list_collections(
    app_name: str,
    system_store: SystemStoreDep,
) -> CollectionsResponse:
    """List all structured and vector collections registered for an application."""
    record = await system_store.get_app(app_name)
    if record is None or record.status != "active":
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found or not active")
    config = AppConfig.from_yaml(record.config_yaml)
    structured = [sc.name for sc in config.structured_collections]
    vector = [vc.name for vc in config.vector_collections]
    return CollectionsResponse(structured=structured, vector=vector)


@router.post("/{app_name}/collections/{collection}/query", response_model=CollectionQueryResponse)
async def query_collection(
    app_name: str,
    collection: str,
    body: CollectionQueryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> CollectionQueryResponse:
    """Query a collection directly, bypassing the LLM agent loop.

    Structured collections support field filtering and field selection.
    Vector collections do not yet support direct querying.
    """
    record = await system_store.get_app(app_name)
    if record is None or record.status != "active":
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found or not active")
    config = AppConfig.from_yaml(record.config_yaml)
    sc_names = {sc.name for sc in config.structured_collections}
    vc_names = {vc.name for vc in config.vector_collections}

    if collection in sc_names:
        app = await _get_active_app(app_name, app_cache, system_store, system_resources)
        runner = app.query_runner
        filters = [_to_filter(f) for f in body.filters]
        records = await runner.structured_store.query(collection, filters or None, body.fields or None)
        return CollectionQueryResponse(collection=collection, records=records, total=len(records))

    if collection in vc_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{collection}' is a vector collection; direct querying is not yet supported. Use POST /{app_name}/query instead.",
        )

    raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")


# ---------------------------------------------------------------------------
# Doc registry endpoints
# ---------------------------------------------------------------------------


@router.get("/{app_name}/docs", response_model=DocListResponse)
async def list_docs(
    app_name: str,
    system_store: SystemStoreDep,
    status: str | None = None,
) -> DocListResponse:
    """List all documents ingested into an application.

    Filter by status: 'active', 'failed', or 'deleted'.
    """
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    docs = await system_store.list_docs(app_name, status=status)
    items = [_doc_to_response(d) for d in docs]
    return DocListResponse(docs=items, total=len(items))


@router.get("/{app_name}/docs/{doc_id}", response_model=DocResponse)
async def get_doc(
    app_name: str,
    doc_id: str,
    system_store: SystemStoreDep,
) -> DocResponse:
    """Return the registry record for a single ingested document."""
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    doc = await system_store.get_doc(app_name, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return _doc_to_response(doc)


@router.delete("/{app_name}/docs/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_doc(
    app_name: str,
    doc_id: str,
    system_store: SystemStoreDep,
) -> None:
    """Remove a document from the registry and cascade-delete its workflow tasks.

    Note: this does not yet remove data from vector or structured stores.
    """
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    doc = await system_store.get_doc(app_name, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    await system_store.delete_doc(app_name, doc_id)
    logger.info("Document '%s' deleted from app '%s'", doc_id, app_name)


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------


@router.get("/{app_name}/tasks", response_model=TaskListResponse)
async def list_tasks(
    app_name: str,
    system_store: SystemStoreDep,
    task_type: str | None = None,
    task_name: str | None = None,
    doc_id: str | None = None,
    status: str | None = None,
) -> TaskListResponse:
    """List background tasks for an application.

    Filter by task_type ('ingest' or 'workflow'), task_name (workflow name),
    doc_id, or status ('pending', 'running', 'done', 'failed').
    """
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    tasks = await system_store.list_tasks(
        app_name, task_type=task_type, task_name=task_name, doc_id=doc_id, status=status
    )
    items = [TaskResponse(**t.model_dump()) for t in tasks]
    return TaskListResponse(tasks=items, total=len(items))


@router.get("/{app_name}/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    app_name: str,
    task_id: str,
    system_store: SystemStoreDep,
) -> TaskResponse:
    """Return a single task by ID."""
    task = await system_store.get_task(task_id)
    if task is None or task.app_name != app_name:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TaskResponse(**task.model_dump())


# ---------------------------------------------------------------------------
# Workflow endpoints
# ---------------------------------------------------------------------------


@router.get("/{app_name}/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    app_name: str,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> WorkflowListResponse:
    """List all workflows registered for an application."""
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    return WorkflowListResponse(app_name=app_name, workflows=app.workflows)



@router.get("/{app_name}/workflows/{workflow_name}/docs", response_model=WorkflowDocListResponse)
async def list_workflow_docs(
    app_name: str,
    workflow_name: str,
    system_store: SystemStoreDep,
    doc_status: str | None = None,
) -> WorkflowDocListResponse:
    """List documents relative to a workflow.

    doc_status options:
    - 'unprocessed': docs with no completed workflow task for this workflow
    - 'done': docs that have a completed workflow task
    - 'failed': docs whose last workflow task failed
    - omit to return all docs with any workflow task record
    """
    record = await system_store.get_app(app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    if doc_status == "unprocessed":
        all_docs = await system_store.list_docs(app_name, status="active")
        done_tasks = await system_store.list_tasks(
            app_name, task_type="workflow", task_name=workflow_name, status="done"
        )
        processed_ids = {t.doc_id for t in done_tasks if t.doc_id}
        docs = [d for d in all_docs if d.doc_id not in processed_ids]
    else:
        all_docs = await system_store.list_docs(app_name, status="active")
        workflow_tasks = await system_store.list_tasks(
            app_name, task_type="workflow", task_name=workflow_name, status=doc_status
        )
        matched_ids = {t.doc_id for t in workflow_tasks if t.doc_id}
        docs = [d for d in all_docs if d.doc_id in matched_ids]

    items = [_doc_to_response(d) for d in docs]
    return WorkflowDocListResponse(
        app_name=app_name,
        workflow_name=workflow_name,
        docs=items,
        total=len(items),
    )


@router.post("/{app_name}/workflows/{workflow_name}/stream")
async def stream_workflow(
    app_name: str,
    workflow_name: str,
    body: WorkflowRunRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> StreamingResponse:
    """Stream workflow results as Server-Sent Events.

    Each saved record yields: ``{"record": {...}}``
    Sentinel: ``data: [DONE]``
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    try:
        wf_runner = app.get_workflow(workflow_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

    pending = await system_store.list_tasks(
        app_name, task_type="workflow", task_name=workflow_name, doc_id=body.doc_id, status="pending"
    )
    if not pending:
        params_list = await app.resolve_workflow_params(wf_runner, body.doc_id)
        pending = [
            await system_store.get_task(
                await system_store.create_workflow_task(app_name, workflow_name, body.doc_id, json.dumps(p))
            )
            for p in params_list
        ]

    async def event_stream():
        for task in pending:
            params = json.loads(task.params_json) if task.params_json else {}
            await system_store.update_task(task.task_id, status="running")
            try:
                async for record in wf_runner.run(params):
                    yield f"data: {json.dumps({'record': record})}\n\n"
                await system_store.complete_workflow_task(task.task_id, success=True)
            except Exception as exc:
                logger.exception("stream_workflow failed app=%s workflow=%s task=%s", app_name, workflow_name, task.task_id)
                await system_store.complete_workflow_task(task.task_id, success=False, error=str(exc))
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
