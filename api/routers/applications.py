"""CRUD endpoints for managing CogBase applications."""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timezone

import json
import yaml

logger = logging.getLogger(__name__)
from fastapi import APIRouter, File, HTTPException, UploadFile, status
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
    CollectionsResponse,
    FilterRequest,
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    IngestResultResponse,
    QueryRequest,
    QueryResponse,
    CollectionQueryRequest,
    CollectionQueryResponse,
    WorkflowListResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from api.system_store import AppRecord, SystemStore
from cogbase.core.models import Document
from cogbase.stores.filters import Filter, Op

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
    # by_alias=True preserves "schema" as the YAML key (field name is schema_).
    stored_yaml = yaml.dump(config.model_dump(by_alias=True), allow_unicode=True, default_flow_style=False)
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


def _serialize_config(config: AppConfig) -> str:
    return yaml.dump(config.model_dump(by_alias=True), allow_unicode=True, default_flow_style=False)


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
    app = await build_app(config, system=system_resources, app_status=record.status)
    app_cache.add(app_name, app)
    return app


# TODO ingest a list of documents may run a long time, make it a background task,
#      client checks and waits for task to complete.
@router.post("/{app_name}/ingest_documents", response_model=IngestDocumentsResponse)
async def ingest_documents(
    app_name: str,
    body: IngestDocumentsRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> IngestDocumentsResponse:
    """Ingest a batch of documents into an active application.

    Documents are processed concurrently up to *concurrency* at a time.  A
    failure on one document does not abort the others — each result carries
    ``success`` and ``error`` for per-document reporting.
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    documents = [Document(doc_id=d.doc_id, text=d.text, metadata=d.metadata) for d in body.documents]
    try:
        results = await app.ingest_documents(documents, concurrency=body.concurrency)
    except Exception:
        logger.exception("ingest_documents failed for app '%s', retrying with fresh app", app_name)
        app = await _get_active_app(
            app_name, app_cache, system_store, system_resources, force_refresh=True
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
    """Drain app.query_stream and return the final result."""
    async for item in app.query_stream(text):
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
    try:
        result = await _drain_query(app, body.text)
    except Exception:
        logger.exception("query failed for app '%s', retrying with fresh app", app_name)
        app = await _get_active_app(
            app_name, app_cache, system_store, system_resources, force_refresh=True
        )
        result = await _drain_query(app, body.text)
    return QueryResponse(
        answer=result.answer,
        passthrough=result.passthrough,
        structured_records=result.structured_records,
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
    Final event:  ``{"result": {answer, passthrough, structured_records}}``
    Sentinel:     ``data: [DONE]``
    """
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)

    async def event_stream():
        try:
            async for item in app.query_stream(body.text):
                if isinstance(item, str):
                    yield f"data: {json.dumps({'token': item})}\n\n"
                else:
                    payload = {
                        "result": {
                            "answer": item.answer,
                            "passthrough": item.passthrough,
                            "structured_records": item.structured_records,
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
            update={"config_yaml": _serialize_config(updated_config), "updated_at": _now()}
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
        update={"config_yaml": _serialize_config(updated_config), "updated_at": _now()}
    )
    await system_store.save_app(updated_record)
    logger.info("Removed skill '%s' from application '%s'", skill_name, app_name)


@router.get("/{app_name}/collections", response_model=CollectionsResponse)
async def list_collections(
    app_name: str,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> CollectionsResponse:
    """List all structured and vector collections registered for an application."""
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    runner = app.query_runner

    structured: list[str] = []
    if runner.structured_store is not None:
        structured = await runner.structured_store.list_collections()

    vector: list[str] = []
    if runner.vector_store is not None:
        vector = await runner.vector_store.list_collections()

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
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    runner = app.query_runner

    if runner.structured_store is not None:
        if collection in await runner.structured_store.list_collections():
            filters = [_to_filter(f) for f in body.filters]
            records = await runner.structured_store.query(collection, filters or None, body.fields or None)
            return CollectionQueryResponse(collection=collection, records=records, total=len(records))

    if runner.vector_store is not None:
        if collection in await runner.vector_store.list_collections():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{collection}' is a vector collection; direct querying is not yet supported. Use POST /{app_name}/query instead.",
            )

    raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")


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


# TODO a workflow may run a long time, need to make it a background task,
#      client checks and waits for task to complete.
@router.post("/{app_name}/workflows/{workflow_name}/run", response_model=WorkflowRunResponse)
async def run_workflow(
    app_name: str,
    workflow_name: str,
    body: WorkflowRunRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> WorkflowRunResponse:
    """Run a workflow and return all saved records when it completes."""
    app = await _get_active_app(app_name, app_cache, system_store, system_resources)
    try:
        wf_runner = app.get_workflow(workflow_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

    records: list[dict] = []
    try:
        async for record in wf_runner.run(body.params):
            records.append(record)
    except Exception as exc:
        logger.exception("run_workflow failed app=%s workflow=%s", app_name, workflow_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return WorkflowRunResponse(workflow=workflow_name, records=records, total=len(records))


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

    async def event_stream():
        try:
            async for record in wf_runner.run(body.params):
                yield f"data: {json.dumps({'record': record})}\n\n"
        except Exception:
            logger.exception("stream_workflow failed app=%s workflow=%s", app_name, workflow_name)
            yield f"data: {json.dumps({'error': 'workflow stream failed'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
