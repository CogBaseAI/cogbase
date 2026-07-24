"""CRUD endpoints for managing CogBase applications."""

from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import urllib.parse
import uuid
import zipfile
from datetime import datetime, timezone

import json
import yaml

logger = logging.getLogger(__name__)
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from cogbase.config.config import AppConfig
from cogbase.stores import AppScope, build_document_store, build_structured_store, build_vector_store
from api.dependencies import (
    AccountIdDep,
    AppCacheDep,
    RequestScopeDep,
    SkillRegistryDep,
    SystemResourcesDep,
    SystemStoreDep,
)
from api.system_resources import SystemResources
from api.factory import build_app
from api.app_cache import AppCache, cache_key
from api.models import (
    AddMemoryRequest,
    AddMemoryResponse,
    AddSkillRequest,
    AppSkillRef,
    AnswerReferences,
    AppSkillsResponse,
    ApplicationListResponse,
    ApplicationResponse,
    ChunkResponse,
    DocumentSliceResponse,
    CollectionsResponse,
    DocListResponse,
    DocResponse,
    DocWorkflowResponse,
    FilterRequest,
    IngestDocumentsAcceptedResponse,
    IngestResultSummary,
    MemoryListResponse,
    MemoryRecordResponse,
    MemoryReviewRequest,
    MemoryReviewResponse,
    MemoryReviewResultItem,
    PendingMemoriesResponse,
    QueryMemoryResponse,
    QueryRequest,
    QueryResponse,
    SessionStartRequest,
    SessionResponse,
    SessionCloseResponse,
    SessionDeleteResponse,
    SessionSummary,
    SessionListResponse,
    SessionTranscriptResponse,
    TranscriptMessage,
    CollectionQueryRequest,
    CollectionQueryResponse,
    TaskListResponse,
    TaskResponse,
    TaskSummaryResponse,
    WorkflowDocListResponse,
    WorkflowListResponse,
    WorkflowRunRequest,
)
from api.system_store import (
    AppRecord, DocRecord, DocWorkflowRecord, DocWorkflowStatus,
    SystemStore, TaskRecord, TaskStatus, new_app_id,
)
from cogbase.stores.filters import Filter, Op
from api.task_runner import DEFAULT_TASK_CONCURRENCY, run_distill_task, run_ingest_task

# Name-addressed routes live under a namespace path segment: an app's client-facing
# name is only unique within (account, namespace). The account is the X-Account-Id
# header (see RequestScopeDep); the namespace is the {namespace} path param.
router = APIRouter(prefix="/namespaces/{namespace}/applications", tags=["applications"])

# Account-wide routes (no namespace segment) — e.g. list every app in the account.
account_router = APIRouter(prefix="/applications", tags=["applications"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _doc_to_response(record: DocRecord, app_name: str) -> DocResponse:
    import json as _json
    try:
        meta = _json.loads(record.metadata) if record.metadata else {}
    except Exception:
        meta = {}
    return DocResponse(
        doc_id=record.doc_id,
        app_name=app_name,
        status=record.status,
        ingested_at=record.ingested_at,
        metadata=meta,
    )


def _doc_workflow_to_response(record: DocRecord, workflow_status: str, app_name: str) -> DocWorkflowResponse:
    import json as _json
    try:
        meta = _json.loads(record.metadata) if record.metadata else {}
    except Exception:
        meta = {}
    return DocWorkflowResponse(
        doc_id=record.doc_id,
        app_name=app_name,
        status=record.status,
        ingested_at=record.ingested_at,
        metadata=meta,
        workflow_status=workflow_status,
    )


def _task_to_response(record: TaskRecord, app_name: str) -> TaskResponse:
    """Map a TaskRecord (id-keyed) to a name-facing TaskResponse."""
    result = None
    if record.result_json:
        try:
            result = IngestResultSummary.model_validate_json(record.result_json)
        except Exception:
            logger.warning("task %s has unparseable result_json", record.task_id)
    return TaskResponse(
        app_name=app_name,
        result=result,
        **record.model_dump(exclude={"app_id", "result_json"}),
    )


def _to_response(record: AppRecord) -> ApplicationResponse:
    try:
        config_dict = yaml.safe_load(record.config_yaml) or {}
    except Exception:
        config_dict = {}
    return ApplicationResponse(
        name=record.name,
        account_id=record.account_id,
        namespace=record.namespace_id,
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


def _validate_skills(skill_ids: list[str], skill_registry) -> None:
    """Raise HTTP 422 if any skill id is not in the registry."""
    unknown = []
    for skill_id in skill_ids:
        try:
            skill_registry.get(skill_id)
        except KeyError:
            unknown.append(skill_id)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown skill id(s): {', '.join(unknown)}. Run GET /skills to see available skills.",
        )


def _app_skills_response(app_name: str, skill_ids: list[str], skill_registry) -> AppSkillsResponse:
    """Build an AppSkillsResponse, resolving display names from the registry.

    A referenced skill normally always exists — ``delete_skill`` refuses to remove
    a skill still referenced by an application (the 409 guard). But that invariant
    can be broken out of band: a skill dropped from ``skills_dir``, or a freshly
    started node whose registry has not yet finished syncing from the system store.
    Resolve each id defensively so one dangling reference can't 500 the whole
    listing — a ref that no longer maps to a live skill is surfaced as ``missing``
    (rather than dropped) so the UI can show it as broken and offer to unassign it.
    """
    refs: list[AppSkillRef] = []
    for skill_id in skill_ids:
        try:
            skill = skill_registry.get(skill_id)
        except KeyError:
            logger.warning(
                "Application '%s' references unknown skill id '%s'; marking as missing",
                app_name,
                skill_id,
            )
            refs.append(AppSkillRef(id=skill_id, name=skill_id, missing=True))
            continue
        refs.append(AppSkillRef(id=skill_id, name=skill.name))
    return AppSkillsResponse(app_name=app_name, skills=refs)


@router.post("", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
async def create_application(
    scope: RequestScopeDep,
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

    if await system_store.get_app(scope.account_id, scope.namespace_id, config.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application '{config.name}' already exists",
        )

    # The namespace must be created explicitly (POST /namespaces) before it can
    # hold applications — there is no implicit landing namespace.
    if await system_store.get_namespace(scope.account_id, scope.namespace_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Namespace '{scope.namespace_id}' does not exist; "
                "create it via POST /namespaces before creating an application"
            ),
        )

    now = _now()
    app_id = new_app_id()
    record = AppRecord(
        app_id=app_id,
        account_id=scope.account_id,
        namespace_id=scope.namespace_id,
        name=config.name,
        config_yaml=yaml_text,
        status="initializing",
        created_at=now,
        updated_at=now,
    )
    await system_store.save_app(record)
    logger.info(
        "Creating application '%s' (app_id=%s account=%s namespace=%s)",
        config.name, app_id, scope.account_id, scope.namespace_id,
    )

    try:
        app = await build_app(
            config, app_id=app_id, account_id=scope.account_id,
            namespace_id=scope.namespace_id, system=system_resources,
            app_status=record.status, task_store=system_store,
        )
        app_cache.add(cache_key(scope.account_id, scope.namespace_id, config.name), app)
        record = record.model_copy(update={"status": "active", "updated_at": _now()})
        logger.info("Application '%s' created successfully", config.name)
    except Exception as exc:
        logger.exception("Failed to create application '%s'", config.name)
        record = record.model_copy(
            update={"status": "error", "error": str(exc), "updated_at": _now()}
        )

    await system_store.save_app(record)
    return _to_response(record)


@account_router.get("", response_model=ApplicationListResponse)
async def list_account_applications(
    account_id: AccountIdDep,
    system_store: SystemStoreDep,
) -> ApplicationListResponse:
    """Return every application in the calling account, across all namespaces."""
    records = await system_store.list_apps(account_id)
    items = [_to_response(r) for r in records]
    logger.info("list account=%s apps=%d", account_id, len(items))
    return ApplicationListResponse(applications=items, total=len(items))


@router.get("", response_model=ApplicationListResponse)
async def list_applications(
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
) -> ApplicationListResponse:
    """Return the applications in one namespace of the calling account."""
    records = await system_store.list_apps(scope.account_id, scope.namespace_id)
    items = [_to_response(r) for r in records]
    logger.info("list account=%s namespace=%s apps=%d", scope.account_id, scope.namespace_id, len(items))
    return ApplicationListResponse(applications=items, total=len(items))


@router.get("/{app_name}", response_model=ApplicationResponse)
async def get_application(
    app_name: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
) -> ApplicationResponse:
    """Return metadata for a single application."""
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    return _to_response(record)


@router.patch("/{app_name}", response_model=ApplicationResponse)
async def update_application(
    app_name: str,
    scope: RequestScopeDep,
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
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    yaml_text, config = _parse_bundle(await bundle.read())

    if config.skills:
        _validate_skills(config.skills, skill_registry)

    if config.name != app_name and await system_store.get_app(scope.account_id, scope.namespace_id, config.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application '{config.name}' already exists",
        )

    app_cache.remove(cache_key(scope.account_id, scope.namespace_id, app_name))

    # The record's identity is its stable app_id, so a rename just updates the
    # ``name`` field on the same row — storage keyed by app_id never moves.
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
    logger.info("Updating application '%s' (app_id=%s)", app_name, record.app_id)

    try:
        app = await build_app(
            config, app_id=record.app_id, account_id=record.account_id,
            namespace_id=record.namespace_id, system=system_resources,
            app_status=updated.status, task_store=system_store,
        )
        app_cache.add(cache_key(scope.account_id, scope.namespace_id, config.name), app)
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
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    system_resources: SystemResourcesDep,
) -> None:
    """Permanently remove an application and its metadata."""
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    config = AppConfig.from_yaml(record.config_yaml)
    app_scope = AppScope(
        account_id=record.account_id, namespace_id=record.namespace_id, app_id=record.app_id
    )

    vector_store = (
        build_vector_store(config.vector_store, scope=app_scope)
        if config.vector_store
        else (system_resources.vector_store.with_scope(app_scope) if system_resources.vector_store else None)
    )
    if vector_store:
        for vc_cfg in config.vector_collections:
            try:
                await vector_store.delete_collection(vc_cfg.name)
                logger.info("Deleted vector collection '%s' for app '%s'", vc_cfg.name, app_name)
            except Exception:
                logger.warning(
                    "Failed to delete vector collection '%s' for app '%s'",
                    vc_cfg.name, app_name, exc_info=True,
                )

    structured_store = (
        build_structured_store(config.structured_store, scope=app_scope)
        if config.structured_store
        else (system_resources.structured_store.with_scope(app_scope) if system_resources.structured_store else None)
    )
    if structured_store:
        for sc_cfg in config.structured_collections:
            try:
                await structured_store.delete_collection(sc_cfg.name)
                logger.info("Deleted structured collection '%s' for app '%s'", sc_cfg.name, app_name)
            except Exception:
                logger.warning(
                    "Failed to delete structured collection '%s' for app '%s'",
                    sc_cfg.name, app_name, exc_info=True,
                )

    document_store = (
        build_document_store(config.document_store, scope=app_scope)
        if config.document_store
        else (system_resources.document_store.with_scope(app_scope) if system_resources.document_store else None)
    )
    if document_store:
        try:
            await document_store.delete_collection(record.app_id)
            logger.info("Deleted document store collection for app '%s'", app_name)
        except Exception:
            logger.warning(
                "Failed to delete document store collection for app '%s'",
                app_name, exc_info=True,
            )

    app_cache.remove(cache_key(scope.account_id, scope.namespace_id, app_name))
    await system_store.delete_app(record.app_id)
    logger.info("Application '%s' deleted", app_name)


async def _get_active_app(
    account_id: str,
    namespace_id: str,
    app_name: str,
    app_cache: AppCache,
    system_store: SystemStore,
    system_resources: SystemResources,
    *,
    force_refresh: bool = False,
) -> object:
    key = cache_key(account_id, namespace_id, app_name)
    if not force_refresh:
        app = app_cache.get(key)
        if app is not None:
            return app
    else:
        app_cache.remove(key)
    record = await system_store.get_app(account_id, namespace_id, app_name)
    if record is None or record.status != "active":
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found or not active")
    config = AppConfig.from_yaml(record.config_yaml)
    app = await build_app(
        config, app_id=record.app_id, account_id=record.account_id,
        namespace_id=record.namespace_id, system=system_resources,
        app_status=record.status, task_store=system_store,
    )
    app_cache.add(key, app)
    return app


@router.post("/{app_name}/upload_documents", response_model=IngestDocumentsAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    app_name: str,
    scope: RequestScopeDep,
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

    Each file is saved to the document store immediately, then an ingestion task
    is created (pending).  The background task parses the file to markdown and
    runs the ingestion pipeline.  Returns task IDs immediately (HTTP 202).
    Poll GET /{app_name}/tasks?task_type=ingest.
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

    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    # Identity for all storage keys comes from the persisted record, not the app
    # instance — the stable app_id, not the (mutable) client-facing name.
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    app_id = record.app_id

    def _safe_doc_id(filename: str) -> str:
        stem = pathlib.Path(filename).stem
        return re.sub(r"[^\w\-]", "_", stem)

    now = _now()
    batch_id = str(uuid.uuid4())
    all_task_ids: list[str] = []
    pending_task_ids: list[str] = []

    for upload in files:
        filename = upload.filename or "upload"
        suffix = pathlib.Path(filename).suffix.lower()
        doc_id = _safe_doc_id(filename)
        task_id = str(uuid.uuid4())
        all_task_ids.append(task_id)
        content = await upload.read()
        doc_metadata = {"source_filename": filename, "source_format": suffix.lstrip("."), **extra_metadata}
        doc_path = f"originals/{doc_id}{suffix}"

        # 1. Save document bytes to document store.
        try:
            await app.document_store.save_bytes(app_id, doc_path, content)
        except NotImplementedError:
            logger.warning("upload_documents: save_bytes not supported, skipping raw save doc_id=%s", doc_id)
        except Exception:
            logger.exception("upload_documents: failed to save original doc_id=%s", doc_id)

        # 2. Create ingestion task with doc_path + doc_metadata in params_json so the
        #    task is self-contained: if the node crashes and the task is retried, it can
        #    reconstruct everything it needs from the task record + document store alone.
        await system_store.create_task(TaskRecord(
            task_id=task_id,
            account_id=record.account_id,
            namespace_id=record.namespace_id,
            app_id=app_id,
            task_type="ingest",
            task_name="ingest",
            doc_id=doc_id,
            batch_id=batch_id,
            params_json=json.dumps({"doc_path": doc_path, "doc_metadata": doc_metadata}),
            status=TaskStatus.PENDING,
            created_at=now,
        ))
        pending_task_ids.append(task_id)

    # 3. Background: for each pending task load bytes from document store, parse,
    #    ingest. Execution is shared with the startup recovery sweep (see
    #    api/task_runner.py) so an interrupted upload is requeued on restart.
    if pending_task_ids:
        async def _run_upload_bg() -> None:
            semaphore = asyncio.Semaphore(DEFAULT_TASK_CONCURRENCY)

            async def _ingest_one(task_id: str) -> None:
                async with semaphore:
                    await run_ingest_task(
                        task_id, app=app, app_name=app_name, app_cache=app_cache,
                        app_id=app_id, system_store=system_store,
                    )

            await asyncio.gather(*(_ingest_one(tid) for tid in pending_task_ids))

        asyncio.create_task(_run_upload_bg())

    return IngestDocumentsAcceptedResponse(
        task_ids=all_task_ids, total=len(all_task_ids), batch_id=batch_id
    )


async def _drain_query(
    app, text: str, history: list[dict] | None = None, system_prompt: str | None = None,
    top_k: int = 10, session_id: str | None = None,
):
    """Drain app.query_stream and return the final result."""
    async for item in app.query_stream(
        text, history=history, system_prompt=system_prompt, top_k=top_k,
        session_id=session_id,
    ):
        if not isinstance(item, str):
            return item
    raise RuntimeError("query_stream did not yield a result")


async def _record_session_turn(system_store, app, session_id: str | None, text: str) -> None:
    """Index a completed turn so the session shows up in the history list.

    Best-effort: a failed index write must not fail an otherwise-successful
    query, so it is logged and swallowed.  Only tracked sessions (a caller that
    passed ``session_id``) are indexed — stateless queries never enter the list.
    """
    if not session_id:
        return
    try:
        await system_store.touch_session(app.account_id, app.namespace_id, app.app_id, session_id, text)
    except Exception:
        logger.exception("failed to index session turn for '%s'", session_id)


@router.post("/{app_name}/query", response_model=QueryResponse)
async def query_application(
    app_name: str,
    scope: RequestScopeDep,
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
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    history = [{"role": m.role, "content": m.content} for m in body.history] or None
    try:
        result = await _drain_query(app, body.text, history=history, system_prompt=body.system_prompt, top_k=body.top_k, session_id=body.session_id)
    except Exception:
        logger.exception("query failed for app '%s', retrying with fresh app", app_name)
        app = await _get_active_app(
            scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources, force_refresh=True
        )
        result = await _drain_query(app, body.text, history=history, system_prompt=body.system_prompt, top_k=body.top_k, session_id=body.session_id)
    await _record_session_turn(system_store, app, body.session_id, body.text)
    return QueryResponse(
        answer=result.answer,
        references=_to_answer_references(result),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        session_id=body.session_id,
    )


@router.post("/{app_name}/query/stream")
async def query_application_stream(
    app_name: str,
    scope: RequestScopeDep,
    body: QueryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> StreamingResponse:
    """Stream a natural-language query response as Server-Sent Events.

    Token events: ``{"token": "<text>"}``
    Final event:  ``{"result": {answer, passthrough, references, input_tokens, output_tokens}}``
                  where ``references`` is the shared ``AnswerReferences`` shape
                  (structured_records, chunks, document_slices, memories).
    Sentinel:     ``data: [DONE]``
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    history = [{"role": m.role, "content": m.content} for m in body.history] or None

    async def event_stream():
        try:
            async for item in app.query_stream(body.text, history=history, system_prompt=body.system_prompt, top_k=body.top_k, session_id=body.session_id):
                if isinstance(item, str):
                    yield f"data: {json.dumps({'token': item})}\n\n"
                else:
                    payload = {
                        "result": {
                            "answer": item.answer,
                            "passthrough": item.passthrough,
                            "references": _to_answer_references(item).model_dump(),
                            "input_tokens": item.input_tokens,
                            "output_tokens": item.output_tokens,
                        }
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    # The result is the last non-token item; index the turn once
                    # it has been produced so the session enters the history list.
                    await _record_session_turn(system_store, app, body.session_id, body.text)
        except Exception:
            logger.exception("query_stream failed for app '%s'", app_name)
            yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Session lifecycle endpoints (short-term + long-term memory)
# ---------------------------------------------------------------------------


@router.post("/{app_name}/sessions", response_model=SessionResponse)
async def start_session(
    app_name: str,
    scope: RequestScopeDep,
    body: SessionStartRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> SessionResponse:
    """Open (or resume) a conversation session and return its id.

    Seeds the short-term metadata cache; the conversational thread itself lives
    in the episodic log and is materialised on the first query.
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    try:
        session_id = await app.start_session(
            metadata=body.metadata,
            session_id=body.session_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return SessionResponse(session_id=session_id)


@router.get("/{app_name}/sessions", response_model=SessionListResponse)
async def list_sessions(
    app_name: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> SessionListResponse:
    """List an app's conversation sessions, most-recently-active first.

    Served entirely from the session index (one query), so the history sidebar
    never replays episodic logs; a session's actual messages are loaded on demand
    via ``GET /{app_name}/sessions/{session_id}``.
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    records = await system_store.list_session_records(app.app_id)
    return SessionListResponse(
        sessions=[
            SessionSummary(
                session_id=r.session_id,
                title=r.title,
                message_count=r.message_count,
                status=r.status,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in records
        ]
    )


@router.get("/{app_name}/sessions/{session_id}", response_model=SessionTranscriptResponse)
async def get_session_transcript(
    app_name: str,
    session_id: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> SessionTranscriptResponse:
    """Return a session's full conversation transcript from the episodic log."""
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    try:
        messages = await app.get_session_transcript(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return SessionTranscriptResponse(
        session_id=session_id,
        messages=[
            TranscriptMessage(
                role=m.role.value,
                content=m.content,
                references=(
                    AnswerReferences.model_validate(m.references) if m.references else None
                ),
            )
            for m in messages
        ],
    )


@router.delete("/{app_name}/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    app_name: str,
    session_id: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> SessionDeleteResponse:
    """Permanently delete a session: drop its episodic log and history-index row.

    Evicts the short-term cache, erases the durable episodic log, then removes the
    session's row from the history index so it disappears from the sidebar.  Any
    long-term memory already distilled from the session is left intact.
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    await app.delete_session(session_id)
    await system_store.delete_session_record(app.app_id, session_id)
    return SessionDeleteResponse(session_id=session_id, deleted=True)


@router.post("/{app_name}/sessions/{session_id}/close", response_model=SessionCloseResponse)
async def close_session(
    app_name: str,
    session_id: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> SessionCloseResponse:
    """Settle a session: evict the short-term cache and enqueue distillation.

    Distillation runs offline (a background task, mirroring the ingestion task
    model) so close returns immediately.  When no distiller is wired the cache
    is still evicted and distillation is reported 'skipped'.
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    await app.end_session(session_id)
    # Flip the session's history-index row to 'closed' (no-op if it never took a turn).
    await system_store.close_session_record(app.app_id, session_id)

    distiller = app.distiller
    if distiller is None:
        return SessionCloseResponse(session_id=session_id, distillation="skipped")

    task_id = await system_store.create_distill_task(app.account_id, app.namespace_id, app.app_id, session_id)

    # Shared with the startup recovery sweep so an interrupted distillation is
    # requeued on restart (see api/task_runner.py).
    asyncio.create_task(run_distill_task(task_id, app=app, system_store=system_store))
    return SessionCloseResponse(
        session_id=session_id, distillation="enqueued", task_id=task_id
    )


# ---------------------------------------------------------------------------
# Long-term memory add endpoint (ingest a conversation into memory)
# ---------------------------------------------------------------------------


@router.post("/{app_name}/memory", response_model=AddMemoryResponse)
async def add_memory(
    app_name: str,
    scope: RequestScopeDep,
    body: AddMemoryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> AddMemoryResponse:
    """Add conversation messages to long-term memory and return what was distilled.

    A self-contained "add memory" call (mem0's ``add`` shape): the batch is
    appended to a session's episodic log, distilled into durable facts, and
    everything distilled is activated so it is immediately recallable — no
    separate session-close or review step.  ``session_id`` is optional; a fresh
    one is generated and returned when omitted.
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    try:
        session_id, records = await app.add_memory(
            messages=[m.model_dump() for m in body.messages],
            session_id=body.session_id,
            metadata=body.metadata,
            observation_date=body.observation_date,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return AddMemoryResponse(
        session_id=session_id,
        memories=[_to_query_memory(r) for r in records],
    )


# ---------------------------------------------------------------------------
# Long-term memory review endpoints (the pending_review -> active gate)
# ---------------------------------------------------------------------------


def _to_memory_response(record) -> MemoryRecordResponse:
    """Serialize a ``LongTermRecord`` (provenance included) for a reviewer."""
    return MemoryRecordResponse.model_validate(record.model_dump(mode="json"))


def _to_query_memory(record) -> QueryMemoryResponse:
    """Project a ``LongTermRecord`` the answer drew on for a query response."""
    return QueryMemoryResponse(
        memory_id=record.memory_id,
        kind=record.kind.value,
        content=record.content,
        entities=list(record.entities),
    )


def _to_answer_references(result) -> AnswerReferences:
    """Project a runner ``QueryResult``'s evidence into the shared references shape.

    The single builder both the blocking and streaming query endpoints use, so a
    live answer and a replayed transcript turn carry identical references.
    """
    return AnswerReferences(
        structured_records=result.structured_records,
        chunks=[ChunkResponse(**c.model_dump(exclude={"embedding"})) for c in result.chunks],
        document_slices=[DocumentSliceResponse(**s.model_dump()) for s in result.document_slices],
        memories=[_to_query_memory(m) for m in result.memories],
    )


@router.get("/{app_name}/memory", response_model=MemoryListResponse)
async def list_memories(
    app_name: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
    status: str | None = "active",
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> MemoryListResponse:
    """Browse stored long-term memories (most-recently-observed first).

    The inspection surface behind the Memory tab's records view. Defaults to
    ``active`` records (what the query path actually recalls); pass ``status=all``
    to span every lifecycle state, or a specific status / ``kind`` to filter.
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    # `status` is a query param here, shadowing the fastapi `status` module — use
    # literal HTTP codes below rather than `status.HTTP_*`.
    parsed_status = _parse_memory_status(status)
    parsed_kind = _parse_memory_kind(kind)
    try:
        records = await app.memories(
            status=parsed_status, kind=parsed_kind, limit=limit, offset=offset
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    items = [_to_memory_response(r) for r in records]
    return MemoryListResponse(memories=items, total=len(items))


@router.get("/{app_name}/memory/pending", response_model=PendingMemoriesResponse)
async def list_pending_memories(
    app_name: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> PendingMemoriesResponse:
    """List the gated long-term memories awaiting review (oldest first).

    Behaviour-affecting kinds (facts, corrections) are promoted to
    ``pending_review`` and stay out of recall until accepted here.
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    parsed_kind = _parse_memory_kind(kind)
    try:
        records = await app.pending_memories(kind=parsed_kind, limit=limit, offset=offset)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return PendingMemoriesResponse(memories=[_to_memory_response(r) for r in records])


@router.post("/{app_name}/memory/review", response_model=MemoryReviewResponse)
async def review_memories(
    app_name: str,
    scope: RequestScopeDep,
    body: MemoryReviewRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> MemoryReviewResponse:
    """Accept (-> active) or reject (-> superseded) gated memories in one batch.

    Returns a per-item outcome: 'accepted' / 'rejected' / 'skipped' (a record no
    longer pending) / 'not_found'.  The batch is server-capped; an over-cap
    request is rejected with 422.
    """
    from cogbase.memory import ReviewDecision

    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    decisions = [
        ReviewDecision(memory_id=item.memory_id, accept=item.decision == "accept")
        for item in body.decisions
    ]
    try:
        results = await app.review_memories(decisions=decisions)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return MemoryReviewResponse(
        results=[
            MemoryReviewResultItem(memory_id=r.memory_id, outcome=r.outcome.value)
            for r in results
        ]
    )


def _parse_memory_kind(kind: str | None):
    """Resolve an optional ``kind`` query param to a ``MemoryKind`` or 422."""
    if kind is None:
        return None
    from cogbase.memory import MemoryKind

    try:
        return MemoryKind(kind)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid memory kind: {kind!r}",
        )


def _parse_memory_status(status_value: str | None):
    """Resolve an optional ``status`` query param to a ``MemoryStatus`` or 422.

    ``None`` or ``"all"`` mean no status filter (span every lifecycle state).
    """
    if status_value is None or status_value == "all":
        return None
    from cogbase.memory import MemoryStatus

    try:
        return MemoryStatus(status_value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid memory status: {status_value!r}",
        )


# ---------------------------------------------------------------------------
# Application skills endpoints
# ---------------------------------------------------------------------------


@router.get("/{app_name}/skills", response_model=AppSkillsResponse)
async def list_application_skills(
    app_name: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
    skill_registry: SkillRegistryDep,
) -> AppSkillsResponse:
    """Return the skills currently assigned to an application (id + display name)."""
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    config = AppConfig.from_yaml(record.config_yaml)
    return _app_skills_response(app_name, config.skills, skill_registry)


@router.post("/{app_name}/skills", response_model=AppSkillsResponse, status_code=status.HTTP_201_CREATED)
async def add_application_skill(
    app_name: str,
    scope: RequestScopeDep,
    body: AddSkillRequest,
    system_store: SystemStoreDep,
    skill_registry: SkillRegistryDep,
) -> AppSkillsResponse:
    """Assign a system skill to an application by name.

    The skill must exist in the system skill registry (uploaded via ``POST /skills``
    or loaded from ``skills_dir``).  Adding the same skill twice is idempotent.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    try:
        skill = skill_registry.get_by_name(body.skill_name, scope.account_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{body.skill_name}' not found in the system skill registry",
        )

    skill_id = skill.id
    config = AppConfig.from_yaml(record.config_yaml)
    if skill_id not in config.skills:
        updated_config = config.model_copy(update={"skills": config.skills + [skill_id]})
        updated_record = record.model_copy(
            update={"config_yaml": updated_config.to_yaml(), "updated_at": _now()}
        )
        await system_store.save_app(updated_record)
        logger.info("Added skill '%s' (id=%s) to application '%s'", body.skill_name, skill_id, app_name)
        config = updated_config

    return _app_skills_response(app_name, config.skills, skill_registry)


@router.delete("/{app_name}/skills/{skill_ref}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def remove_application_skill(
    app_name: str,
    skill_ref: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
    skill_registry: SkillRegistryDep,
) -> None:
    """Remove a skill from an application by display name or by skill id.

    Live skills are addressed by name (resolved to their id via the registry).
    A dangling reference to a skill that no longer exists in the registry can't
    be resolved by name, so the raw skill id is also accepted — this is what lets
    the UI clean up a ``missing`` ref surfaced by ``_app_skills_response``.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    try:
        skill_id = skill_registry.get_by_name(skill_ref, scope.account_id).id
    except KeyError:
        skill_id = skill_ref  # fall back to treating the ref as a raw skill id (missing skill)

    config = AppConfig.from_yaml(record.config_yaml)
    if skill_id not in config.skills:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{skill_ref}' is not assigned to application '{app_name}'",
        )

    updated_config = config.model_copy(update={"skills": [s for s in config.skills if s != skill_id]})
    updated_record = record.model_copy(
        update={"config_yaml": updated_config.to_yaml(), "updated_at": _now()}
    )
    await system_store.save_app(updated_record)
    logger.info("Removed skill '%s' (id=%s) from application '%s'", skill_ref, skill_id, app_name)


@router.get("/{app_name}/collections", response_model=CollectionsResponse)
async def list_collections(
    app_name: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
) -> CollectionsResponse:
    """List all structured and vector collections registered for an application."""
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
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
    scope: RequestScopeDep,
    body: CollectionQueryRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> CollectionQueryResponse:
    """Query a collection directly, bypassing the LLM agent loop.

    Structured collections support field filtering and field selection.
    Vector collections do not yet support direct querying.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None or record.status != "active":
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found or not active")
    config = AppConfig.from_yaml(record.config_yaml)
    sc_names = {sc.name for sc in config.structured_collections}
    vc_names = {vc.name for vc in config.vector_collections}

    if collection in sc_names:
        app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
        filters = [_to_filter(f) for f in body.filters]
        records = await app.query_runner.query_collection(collection, filters or None, body.fields or None)
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
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
    status: str | None = None,
) -> DocListResponse:
    """List all documents ingested into an application.

    Filter by status: 'active', 'failed', or 'deleted'.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    docs = await system_store.list_docs(record.app_id, status=status)
    items = [_doc_to_response(d, app_name) for d in docs]
    return DocListResponse(docs=items, total=len(items))


@router.get("/{app_name}/docs/{doc_id}", response_model=DocResponse)
async def get_doc(
    app_name: str,
    doc_id: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
) -> DocResponse:
    """Return the registry record for a single ingested document."""
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    doc = await system_store.get_doc(record.app_id, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return _doc_to_response(doc, app_name)


@router.get("/{app_name}/docs/{doc_id}/original")
async def download_original_document(
    app_name: str,
    doc_id: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> StreamingResponse:
    """Stream the raw uploaded file for an ingested document as a download.

    The original bytes are saved at upload time under ``originals/{doc_id}{suffix}``;
    the suffix and download filename are reconstructed from the document's upload
    metadata (``source_format`` / ``source_filename``).
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    doc = await system_store.get_doc(record.app_id, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    meta = json.loads(doc.metadata) if doc.metadata else {}
    source_format = meta.get("source_format")
    suffix = f".{source_format}" if source_format else ""
    filename = meta.get("source_filename") or f"{doc_id}{suffix}"

    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    try:
        data = await app.document_store.load_bytes(record.app_id, f"originals/{doc_id}{suffix}")
    except (KeyError, NotImplementedError):
        raise HTTPException(status_code=404, detail=f"Original file for document '{doc_id}' not found")
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # HTTP headers are latin-1 encoded, so a non-ASCII filename (e.g. CJK) must be
    # sent via RFC 5987's ``filename*`` with a percent-encoded UTF-8 value. Keep an
    # ASCII-only ``filename`` fallback for clients that ignore the extended form.
    ascii_fallback = filename.encode("ascii", "ignore").decode("ascii") or "download"
    utf8_quoted = urllib.parse.quote(filename, safe="")
    content_disposition = (
        f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{utf8_quoted}'
    )
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": content_disposition},
    )


@router.delete("/{app_name}/docs/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_doc(
    app_name: str,
    doc_id: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> None:
    """Delete a document: purge its ingested data, then drop it from the registry.

    Removes the document's vector chunks and structured records from every
    pipeline, its parsed text from the document store, and its raw uploaded file,
    then cascade-deletes its workflow tasks and the registry record.  The store
    purge runs first so a failure there surfaces before the registry entry — the
    only handle on the document — is gone.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    doc = await system_store.get_doc(record.app_id, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    # Purge derived data (vector + structured + parsed text) from every pipeline's
    # stores before touching the registry. Best-effort for the raw original, whose
    # path we reconstruct from the upload metadata (originals/{doc_id}{suffix}).
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    await app.delete_document(doc_id)

    source_format = (json.loads(doc.metadata) if doc.metadata else {}).get("source_format")
    suffix = f".{source_format}" if source_format else ""
    try:
        await app.document_store.delete(record.app_id, f"originals/{doc_id}{suffix}")
    except Exception:
        logger.exception("delete_doc: failed to remove raw original doc_id=%s", doc_id)

    await system_store.delete_doc(record.app_id, doc_id)
    logger.info("Document '%s' deleted from app '%s'", doc_id, app_name)


# ---------------------------------------------------------------------------
# Generated artifact download
# ---------------------------------------------------------------------------
#
# Artifacts (e.g. a merged document) are produced by skills through the query
# runner's ``save_artifact`` tool, which stores them under ``generated/{id}`` in
# the app's document store. This endpoint is the general download side — it is
# agnostic to how the artifact was produced or what domain it belongs to; the
# merge logic itself lives in the edit-docx skill, not here.
#
# The link the runner emits is keyed by the stable ``app_id`` (not the mutable
# client-facing name), so it keeps resolving after a rename.  ``app_id`` is a
# global UUID that spans namespaces, so the route lives on the account-wide router
# (``/applications/{app_id}/…``, matching the runner link,
# ``QueryRunner._artifact_download_path``) rather than under a namespace segment;
# the owning namespace comes from the resolved record.  The account boundary is
# still enforced: a record resolved by id whose account differs from the caller's
# is treated as not found, so one account can't download another's artifacts.


@account_router.get("/{app_ref}/documents/{doc_id}/download")
async def download_generated_document(
    app_ref: str,
    doc_id: str,
    account_id: AccountIdDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> StreamingResponse:
    """Stream a generated artifact (identified by its full ``artifact_id``) as a download.

    ``app_ref`` is the application's stable global ``app_id`` (what the runner emits);
    a name is also accepted as a fallback, resolved within the caller's account.
    """
    record = await system_store.get_app_by_id(app_ref)
    # ``app_id`` is a global UUID, so guard the account boundary explicitly: a
    # record resolved by id must belong to the calling account or it is treated
    # as not found (rather than served cross-tenant).
    if record is not None and record.account_id != account_id:
        record = None
    if record is None:
        # Fallback for a name-keyed link: match within the caller's account
        # (a name is unique per namespace, so pick the first match across them).
        matches = [r for r in await system_store.list_apps(account_id) if r.name == app_ref]
        record = matches[0] if matches else None
    if record is None or record.status != "active":
        raise HTTPException(status_code=404, detail=f"Application '{app_ref}' not found or not active")
    app = await _get_active_app(
        record.account_id, record.namespace_id, record.name,
        app_cache, system_store, system_resources,
    )
    try:
        data = await app.document_store.load_bytes(record.app_id, f"generated/{doc_id}")
    except (KeyError, NotImplementedError):
        raise HTTPException(status_code=404, detail=f"Generated document '{doc_id}' not found")
    media_type = mimetypes.guess_type(doc_id)[0] or "application/octet-stream"
    # HTTP headers are latin-1 encoded, so a non-ASCII filename (e.g. CJK) must be
    # sent via RFC 5987's ``filename*`` with a percent-encoded UTF-8 value. Keep an
    # ASCII-only ``filename`` fallback for clients that ignore the extended form.
    ascii_fallback = doc_id.encode("ascii", "ignore").decode("ascii") or "download"
    utf8_quoted = urllib.parse.quote(doc_id, safe="")
    content_disposition = (
        f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{utf8_quoted}'
    )
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": content_disposition},
    )


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------


@router.get("/{app_name}/tasks", response_model=TaskListResponse)
async def list_tasks(
    app_name: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
    task_type: str | None = None,
    task_name: str | None = None,
    doc_id: str | None = None,
    batch_id: str | None = None,
    status: TaskStatus | None = None,
) -> TaskListResponse:
    """List background tasks for an application.

    Filter by task_type ('ingest' or 'workflow'), task_name (workflow name),
    doc_id, batch_id (an upload batch), or status ('pending', 'running', 'done',
    'failed').  Finished ingest tasks carry a ``result`` with per-document chunk
    and record counts and any warning.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    tasks = await system_store.list_tasks(
        record.app_id, task_type=task_type, task_name=task_name,
        doc_id=doc_id, batch_id=batch_id, status=status,
    )
    items = [_task_to_response(t, app_name) for t in tasks]
    return TaskListResponse(tasks=items, total=len(items))


@router.get("/{app_name}/tasks/summary", response_model=TaskSummaryResponse)
async def task_summary(
    app_name: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
    batch_id: str | None = None,
    task_type: str | None = None,
) -> TaskSummaryResponse:
    """Roll up task status and ingest counts — answers 'did my upload work?'.

    Counts tasks by status and, across finished ingest tasks, totals the chunks
    and records written and the number that ingested nothing (``warnings``).
    Scope to one upload with ``batch_id`` (returned by upload_documents); narrow
    to ingest or workflow tasks with ``task_type``.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")
    tasks = await system_store.list_tasks(
        record.app_id, task_type=task_type, batch_id=batch_id
    )

    by_status = {s: 0 for s in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.DONE, TaskStatus.FAILED)}
    chunks_written = records_extracted = warnings = 0
    for t in tasks:
        by_status[t.status] = by_status.get(t.status, 0) + 1
        if not t.result_json:
            continue
        try:
            summary = IngestResultSummary.model_validate_json(t.result_json)
        except Exception:
            continue
        chunks_written += summary.chunks_written
        records_extracted += summary.records_extracted
        if summary.warning:
            warnings += 1

    return TaskSummaryResponse(
        app_name=app_name,
        batch_id=batch_id,
        total=len(tasks),
        pending=by_status[TaskStatus.PENDING],
        running=by_status[TaskStatus.RUNNING],
        done=by_status[TaskStatus.DONE],
        failed=by_status[TaskStatus.FAILED],
        chunks_written=chunks_written,
        records_extracted=records_extracted,
        warnings=warnings,
    )


@router.get("/{app_name}/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    app_name: str,
    task_id: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
) -> TaskResponse:
    """Return a single task by ID."""
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    task = await system_store.get_task(task_id)
    if record is None or task is None or task.app_id != record.app_id:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return _task_to_response(task, app_name)


# ---------------------------------------------------------------------------
# Workflow endpoints
# ---------------------------------------------------------------------------


@router.get("/{app_name}/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    app_name: str,
    scope: RequestScopeDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> WorkflowListResponse:
    """List all workflows registered for an application."""
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    return WorkflowListResponse(app_name=app_name, workflows=app.workflows)



@router.get("/{app_name}/workflows/{workflow_name}/docs", response_model=WorkflowDocListResponse)
async def list_workflow_docs(
    app_name: str,
    workflow_name: str,
    scope: RequestScopeDep,
    system_store: SystemStoreDep,
    status: DocWorkflowStatus | None = None,
) -> WorkflowDocListResponse:
    """List documents and their workflow processing status.

    status options: 'ready', 'pending', 'running', 'done', 'failed' — omit to return all.
    """
    record = await system_store.get_app(scope.account_id, scope.namespace_id, app_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_name}' not found")

    wf_records = await system_store.list_doc_workflows(
        record.app_id, workflow_name=workflow_name, status=status
    )
    if not wf_records:
        return WorkflowDocListResponse(app_name=app_name, workflow_name=workflow_name, docs=[], total=0)

    doc_ids = {wr.doc_id for wr in wf_records}
    all_docs = await system_store.list_docs(record.app_id, status="active")
    docs_by_id = {d.doc_id: d for d in all_docs if d.doc_id in doc_ids}

    items = [
        _doc_workflow_to_response(docs_by_id[wr.doc_id], wr.status, app_name)
        for wr in wf_records
        if wr.doc_id in docs_by_id
    ]
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
    scope: RequestScopeDep,
    body: WorkflowRunRequest,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
    system_resources: SystemResourcesDep,
) -> StreamingResponse:
    """Stream workflow results as Server-Sent Events.

    Each saved record yields: ``{"record": {...}}``
    Sentinel: ``data: [DONE]``
    """
    app = await _get_active_app(scope.account_id, scope.namespace_id, app_name, app_cache, system_store, system_resources)
    app_id = app.app_id
    try:
        wf_runner = app.get_workflow(workflow_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")

    pending = await system_store.list_tasks(
        app_id, task_type="workflow", task_name=workflow_name, doc_id=body.doc_id, status=TaskStatus.PENDING
    )
    if not pending:
        params_list = await app.resolve_workflow_params(wf_runner, body.doc_id)
        pending = await system_store.create_workflow_tasks(
            app.account_id, app.namespace_id, app_id, workflow_name, body.doc_id, params_list
        )

    async def event_stream():
        all_ok = True
        # Purge this doc's prior output for this workflow before regenerating, so a
        # manual re-run after a re-ingest replaces the doc's findings instead of
        # orphaning resolved ones. Once, before the param-set fan-out below.
        if body.doc_id:
            try:
                await wf_runner.purge_document(body.doc_id)
            except Exception:
                logger.exception(
                    "stream_workflow purge failed app=%s workflow=%s doc_id=%s",
                    app_name, workflow_name, body.doc_id,
                )
        for task in pending:
            params = json.loads(task.params_json) if task.params_json else {}
            await system_store.update_task(task.task_id, status=TaskStatus.RUNNING, started_at=_now())
            try:
                async for record in wf_runner.run(params):
                    yield f"data: {json.dumps({'record': record})}\n\n"
                await system_store.complete_workflow_task(task.task_id, success=True)
            except Exception as exc:
                all_ok = False
                logger.exception("stream_workflow failed app=%s workflow=%s task=%s", app_name, workflow_name, task.task_id)
                await system_store.complete_workflow_task(task.task_id, success=False, error=str(exc))
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        if body.doc_id:
            # TODO if failed, some items such as some clauses in a contract may be successfully processed,
            #      need to clean up the partial results.
            await system_store.upsert_doc_workflow_status(
                app.account_id, app.namespace_id, app_id, body.doc_id, workflow_name,
                DocWorkflowStatus.DONE if all_ok else DocWorkflowStatus.FAILED,
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
