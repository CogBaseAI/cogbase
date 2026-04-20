"""CRUD endpoints for managing CogBase applications."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import yaml
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.config import AppConfig
from api.dependencies import RegistryDep, SystemConfigDep, SystemStoreDep, SystemStructuredStoreDep
from api.factory import build_app
from api.models import ApplicationListResponse, ApplicationResponse
from api.system_store import AppRecord

router = APIRouter(prefix="/applications", tags=["applications"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_response(record: AppRecord) -> ApplicationResponse:
    try:
        config_dict = yaml.safe_load(record.config_yaml) or {}
    except Exception:
        config_dict = {}
    return ApplicationResponse(
        app_id=record.app_id,
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
    registry: RegistryDep,
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

    existing = await system_store.get_app_by_name(config.name)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application '{config.name}' already exists (app_id={existing.app_id})",
        )

    app_id = str(uuid.uuid4())
    now = _now()
    record = AppRecord(
        app_id=app_id,
        name=config.name,
        config_yaml=yaml_text,
        status="initializing",
        created_at=now,
        updated_at=now,
    )
    await system_store.save_app(record)

    try:
        app = build_app(
            config,
            system_structured_store=system_structured_store,
            system_vector_store_cfg=system_config.vector_store,
            app_namespace=config.name,
        )
        await app.setup()
        registry.add(app_id, app)
        record = record.model_copy(update={"status": "active", "updated_at": _now()})
    except Exception as exc:
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


@router.get("/{app_id}", response_model=ApplicationResponse)
async def get_application(
    app_id: str,
    system_store: SystemStoreDep,
) -> ApplicationResponse:
    """Return metadata for a single application."""
    record = await system_store.get_app(app_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_id}' not found")
    return _to_response(record)


@router.patch("/{app_id}", response_model=ApplicationResponse)
async def update_application(
    app_id: str,
    system_store: SystemStoreDep,
    registry: RegistryDep,
    system_config: SystemConfigDep,
    system_structured_store: SystemStructuredStoreDep,
    config_file: UploadFile = File(..., description="Updated YAML config file"),
) -> ApplicationResponse:
    """Replace an application's config and restart it.

    The old instance is torn down before the new config is applied.  If the new
    config fails to initialise the application, the record is kept with
    ``status=error`` so you can inspect and fix the config.
    """
    record = await system_store.get_app(app_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_id}' not found")

    yaml_text, config = _parse_config(await config_file.read())

    if config.name != record.name:
        conflict = await system_store.get_app_by_name(config.name)
        if conflict is not None and conflict.app_id != app_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Application '{config.name}' already exists (app_id={conflict.app_id})",
            )

    registry.remove(app_id)

    updated = record.model_copy(
        update={
            "config_yaml": yaml_text,
            "name": config.name,
            "status": "initializing",
            "error": None,
            "updated_at": _now(),
        }
    )
    await system_store.save_app(updated)

    try:
        app = build_app(
            config,
            system_structured_store=system_structured_store,
            system_vector_store_cfg=system_config.vector_store,
            app_namespace=config.name,
        )
        await app.setup()
        registry.add(app_id, app)
        updated = updated.model_copy(update={"status": "active", "updated_at": _now()})
    except Exception as exc:
        updated = updated.model_copy(
            update={"status": "error", "error": str(exc), "updated_at": _now()}
        )

    await system_store.save_app(updated)
    return _to_response(updated)


@router.delete("/{app_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_application(
    app_id: str,
    system_store: SystemStoreDep,
    registry: RegistryDep,
) -> None:
    """Permanently remove an application and its metadata."""
    record = await system_store.get_app(app_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Application '{app_id}' not found")
    registry.remove(app_id)
    await system_store.delete_app(app_id)
