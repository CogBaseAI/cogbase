"""Endpoints for uploading and managing system-wide skills.

Skills are uploaded as a ZIP bundle (SKILL.md + scripts/assets). The bundle bytes
are persisted in the system document store (the shared, multi-node source of
truth) and materialized into a local cache dir for execution. Each skill gets a
stable UUID; applications reference skills by id so renaming a skill never breaks
existing references.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.dependencies import SkillBundleStoreDep, SkillRegistryDep, SystemStoreDep
from api.models import SkillListResponse, SkillResponse
from api.system_store import SkillRecord
from cogbase.config.config import AppConfig
from cogbase.skills.skill import load_skill_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])


def _to_response(skill) -> SkillResponse:
    return SkillResponse(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        metadata=skill.metadata,
        source_path=str(skill.source_path) if skill.source_path else None,
        builtin=skill.builtin,
    )


def _get_skill_by_name(skill_registry, skill_name: str):
    """Return the skill with *skill_name*, raising HTTP 404 if not found."""
    try:
        return skill_registry.get_by_name(skill_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No skill with name '{skill_name}'")


def _reject_if_builtin(skill) -> None:
    """Block mutating operations on built-in (skills_dir) skills."""
    if skill.builtin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Skill '{skill.name}' is built-in and cannot be updated or deleted.",
        )


async def _ingest_bundle(
    skill_id: str,
    raw: bytes,
    bundle_store,
    skill_registry,
    system_store,
    *,
    replace: bool,
) -> SkillResponse:
    """Materialize, validate, persist, and register a skill bundle."""
    try:
        skill_root = bundle_store.materialize(skill_id, raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid skill bundle: {exc}") from exc

    skill = load_skill_dir(skill_root, skill_id=skill_id)
    if skill is None:
        await bundle_store.delete(skill_id)
        raise HTTPException(
            status_code=422,
            detail="SKILL.md is missing or has invalid YAML front-matter.",
        )

    bundle_key = await bundle_store.save_bundle(skill_id, raw)

    now = datetime.now(timezone.utc).isoformat()
    existing = await system_store.get_skill(skill_id) if replace else None
    record = SkillRecord(
        skill_id=skill_id,
        name=skill.name,
        description=skill.description,
        metadata_json=json.dumps(skill.metadata) if skill.metadata else None,
        bundle_key=bundle_key,
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )
    await system_store.save_skill(record)
    skill_registry.register(skill, replace=replace)
    logger.info("[skills] %s skill id=%s name=%s", "replaced" if replace else "uploaded", skill_id, skill.name)
    return _to_response(skill)


@router.post("", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
async def upload_skill(
    skill_registry: SkillRegistryDep,
    bundle_store: SkillBundleStoreDep,
    system_store: SystemStoreDep,
    bundle: UploadFile = File(..., description="ZIP bundle containing SKILL.md and any scripts/assets"),
) -> SkillResponse:
    """Upload a new skill from a ZIP bundle. Assigns and returns a stable id."""
    raw = await bundle.read()
    skill_id = uuid.uuid4().hex
    return await _ingest_bundle(
        skill_id, raw, bundle_store, skill_registry, system_store, replace=False
    )


@router.put("/{skill_name}", response_model=SkillResponse)
async def replace_skill(
    skill_name: str,
    skill_registry: SkillRegistryDep,
    bundle_store: SkillBundleStoreDep,
    system_store: SystemStoreDep,
    bundle: UploadFile = File(..., description="Updated ZIP bundle containing SKILL.md and any scripts/assets"),
) -> SkillResponse:
    """Replace an existing skill's bundle by name, keeping its id (and so all app references)."""
    skill = _get_skill_by_name(skill_registry, skill_name)
    _reject_if_builtin(skill)
    if await system_store.get_skill(skill.id) is None:
        raise HTTPException(status_code=404, detail=f"No skill with name '{skill_name}'")
    raw = await bundle.read()
    return await _ingest_bundle(
        skill.id, raw, bundle_store, skill_registry, system_store, replace=True
    )


@router.get("", response_model=SkillListResponse)
async def list_skills(skill_registry: SkillRegistryDep) -> SkillListResponse:
    """Return all skills available in the system."""
    items = [_to_response(s) for s in skill_registry.all_skills()]
    return SkillListResponse(skills=items, total=len(items))


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str, skill_registry: SkillRegistryDep) -> SkillResponse:
    """Return a single skill by name."""
    return _to_response(_get_skill_by_name(skill_registry, skill_name))


@router.delete("/{skill_name}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_skill(
    skill_name: str,
    skill_registry: SkillRegistryDep,
    bundle_store: SkillBundleStoreDep,
    system_store: SystemStoreDep,
) -> None:
    """Delete a skill by name from the document store, local cache, and registry.

    A skill that is still assigned to one or more applications cannot be deleted;
    unassign it from those apps first (DELETE /applications/{name}/skills/{skill_name}).
    """
    skill = _get_skill_by_name(skill_registry, skill_name)
    _reject_if_builtin(skill)
    if await system_store.get_skill(skill.id) is None:
        raise HTTPException(status_code=404, detail=f"No skill with name '{skill_name}'")

    # TODO if app count grows large, consider an index of skill_id → apps.
    referencing = []
    for app in await system_store.list_apps():
        if skill.id in AppConfig.from_yaml(app.config_yaml).skills:
            referencing.append(app.name)
    if referencing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Skill '{skill_name}' is still assigned to application(s): "
                f"{', '.join(referencing)}. Unassign it before deleting."
            ),
        )

    await bundle_store.delete(skill.id)
    await system_store.delete_skill(skill.id)
    skill_registry.unregister(skill.id)
    logger.info("[skills] deleted skill name=%s id=%s", skill_name, skill.id)
