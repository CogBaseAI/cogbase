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
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.dependencies import (
    AccountIdDep,
    SkillBundleStoreDep,
    SkillRegistryDep,
    SystemStoreDep,
)
from api.models import (
    SkillContentResponse,
    SkillFile,
    SkillFileResponse,
    SkillListResponse,
    SkillResponse,
)
from api.system_store import SkillRecord
from cogbase.config.config import AppConfig
from cogbase.skills.skill import load_skill_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])

# Cap on how much of a bundle file is returned as text, and the set of
# directories that never carry user-relevant source.
_MAX_FILE_BYTES = 512 * 1024
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}


def _bundle_root(skill) -> Path:
    """Directory holding SKILL.md and any scripts/assets for *skill*."""
    if not skill.source_path:
        raise HTTPException(status_code=404, detail=f"Skill '{skill.name}' has no bundle on disk.")
    return Path(skill.source_path).parent


def _is_text_bytes(raw: bytes) -> bool:
    """Heuristic: treat a file as text if the head has no NUL and decodes as UTF-8."""
    head = raw[:8192]
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _list_bundle_files(root: Path) -> list[SkillFile]:
    """Enumerate the bundle's scripts/assets under *root*, sorted by path.

    The root SKILL.md is omitted — it is returned separately as rendered markdown,
    so listing it here would duplicate it in the UI.
    """
    files: list[SkillFile] = []
    for p in root.rglob("*"):
        rel_parts = p.relative_to(root).parts
        if p.is_dir() or any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if len(rel_parts) == 1 and rel_parts[0] == "SKILL.md":
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        files.append(SkillFile(path=rel, size=len(raw), is_text=_is_text_bytes(raw)))
    return sorted(files, key=lambda f: f.path)


def _resolve_bundle_file(root: Path, rel_path: str) -> Path:
    """Resolve *rel_path* under *root*, rejecting traversal outside the bundle."""
    root = root.resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No file '{rel_path}' in skill bundle.")
    return target


def _to_response(skill) -> SkillResponse:
    return SkillResponse(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        metadata=skill.metadata,
        source_path=str(skill.source_path) if skill.source_path else None,
        builtin=skill.builtin,
    )


def _get_skill_by_name(skill_registry, skill_name: str, account_id: str):
    """Return the account's skill named *skill_name*, raising HTTP 404 if not found."""
    try:
        return skill_registry.get_by_name(skill_name, account_id)
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
    account_id: str,
    *,
    replace: bool,
) -> SkillResponse:
    """Materialize, validate, persist, and register a skill bundle for *account_id*."""
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
        account_id=account_id,
        namespace_id="",  # skills are account-scoped, not per-namespace
        name=skill.name,
        description=skill.description,
        metadata_json=json.dumps(skill.metadata) if skill.metadata else None,
        bundle_key=bundle_key,
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )
    await system_store.save_skill(record)
    skill_registry.register(skill, account_id=account_id, replace=replace)
    logger.info(
        "[skills] %s skill id=%s name=%s account=%s",
        "replaced" if replace else "uploaded", skill_id, skill.name, account_id,
    )
    return _to_response(skill)


@router.post("", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
async def upload_skill(
    account_id: AccountIdDep,
    skill_registry: SkillRegistryDep,
    bundle_store: SkillBundleStoreDep,
    system_store: SystemStoreDep,
    bundle: UploadFile = File(..., description="ZIP bundle containing SKILL.md and any scripts/assets"),
) -> SkillResponse:
    """Upload a new skill from a ZIP bundle. Assigns and returns a stable id."""
    raw = await bundle.read()
    skill_id = uuid.uuid4().hex
    return await _ingest_bundle(
        skill_id, raw, bundle_store, skill_registry, system_store, account_id, replace=False
    )


@router.put("/{skill_name}", response_model=SkillResponse)
async def replace_skill(
    skill_name: str,
    account_id: AccountIdDep,
    skill_registry: SkillRegistryDep,
    bundle_store: SkillBundleStoreDep,
    system_store: SystemStoreDep,
    bundle: UploadFile = File(..., description="Updated ZIP bundle containing SKILL.md and any scripts/assets"),
) -> SkillResponse:
    """Replace an existing skill's bundle by name, keeping its id (and so all app references)."""
    skill = _get_skill_by_name(skill_registry, skill_name, account_id)
    _reject_if_builtin(skill)
    if await system_store.get_skill(skill.id) is None:
        raise HTTPException(status_code=404, detail=f"No skill with name '{skill_name}'")
    raw = await bundle.read()
    return await _ingest_bundle(
        skill.id, raw, bundle_store, skill_registry, system_store, account_id, replace=True
    )


@router.get("", response_model=SkillListResponse)
async def list_skills(
    account_id: AccountIdDep, skill_registry: SkillRegistryDep
) -> SkillListResponse:
    """Return the skills available to the calling account (its own + global builtins)."""
    items = [_to_response(s) for s in skill_registry.all_skills(account_id)]
    return SkillListResponse(skills=items, total=len(items))


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(
    skill_name: str, account_id: AccountIdDep, skill_registry: SkillRegistryDep
) -> SkillResponse:
    """Return a single skill by name."""
    return _to_response(_get_skill_by_name(skill_registry, skill_name, account_id))


@router.get("/{skill_name}/content", response_model=SkillContentResponse)
async def get_skill_content(
    skill_name: str, account_id: AccountIdDep, skill_registry: SkillRegistryDep
) -> SkillContentResponse:
    """Return the full SKILL.md plus a listing of the bundle's scripts/assets.

    Lets the UI show exactly what a skill contains — the LLM-facing markdown and
    the files that ship (and execute) with it.
    """
    skill = _get_skill_by_name(skill_registry, skill_name, account_id)
    return SkillContentResponse(
        id=skill.id,
        name=skill.name,
        markdown=skill.raw_markdown,
        files=_list_bundle_files(_bundle_root(skill)),
    )


@router.get("/{skill_name}/files/{file_path:path}", response_model=SkillFileResponse)
async def get_skill_file(
    skill_name: str, file_path: str, account_id: AccountIdDep, skill_registry: SkillRegistryDep
) -> SkillFileResponse:
    """Return the text content of a single file inside the skill bundle.

    Path-traversal guarded, text-only, and capped at 512 KiB so the UI can let
    users read a skill's scripts without exposing the whole filesystem.
    """
    skill = _get_skill_by_name(skill_registry, skill_name, account_id)
    target = _resolve_bundle_file(_bundle_root(skill), file_path)
    raw = target.read_bytes()
    if not _is_text_bytes(raw):
        raise HTTPException(status_code=415, detail=f"'{file_path}' is not a text file.")
    truncated = len(raw) > _MAX_FILE_BYTES
    content = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
    return SkillFileResponse(
        path=file_path, size=len(raw), truncated=truncated, content=content
    )


@router.delete("/{skill_name}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_skill(
    skill_name: str,
    account_id: AccountIdDep,
    skill_registry: SkillRegistryDep,
    bundle_store: SkillBundleStoreDep,
    system_store: SystemStoreDep,
) -> None:
    """Delete a skill by name from the document store, local cache, and registry.

    A skill that is still assigned to one or more applications cannot be deleted;
    unassign it from those apps first (DELETE /applications/{name}/skills/{skill_name}).
    """
    skill = _get_skill_by_name(skill_registry, skill_name, account_id)
    _reject_if_builtin(skill)
    if await system_store.get_skill(skill.id) is None:
        raise HTTPException(status_code=404, detail=f"No skill with name '{skill_name}'")

    # TODO if app count grows large, consider an index of skill_id → apps.
    # A skill is account-scoped, so only this account's apps can reference it.
    referencing = []
    for app in await system_store.list_apps(account_id):
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
