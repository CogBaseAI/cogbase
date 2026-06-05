"""Durable, multi-node skill persistence.

The document store (e.g. S3) is the shared source of truth: each uploaded skill
is a ZIP bundle stored under ``skills/<skill_id>.zip``. The runner, however, can
only execute scripts from the local filesystem, so bundles are *materialized* into
a local cache dir (``<cache>/<skill_id>/``). A node that has never seen a skill
syncs it from the document store on demand — this is what lets CogBase apps start
on multiple nodes against a shared remote document store.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import zipfile
from pathlib import Path

from cogbase.stores.document.base import DocumentStoreBase

logger = logging.getLogger(__name__)

SKILLS_COLLECTION = "skills"

_SKILLS_CACHE_DIR = os.path.abspath(
    os.environ.get("COGBASE_SKILLS_CACHE_DIR", os.path.expanduser("~/.cogbase/skills"))
)


def bundle_key(skill_id: str) -> str:
    """Document-store key for a skill's ZIP bundle."""
    return f"{skill_id}.zip"


def _find_skill_md(root: Path) -> Path | None:
    """Return the directory containing the shallowest ``SKILL.md`` under *root*."""
    candidates = sorted(root.rglob("SKILL.md"), key=lambda p: len(p.relative_to(root).parts))
    return candidates[0].parent if candidates else None


def _safe_extract(zip_bytes: bytes, dest: Path) -> None:
    """Extract a ZIP archive into *dest*, rejecting entries that escape it (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP entry: {bad}")
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not (target == dest or str(target).startswith(str(dest) + os.sep)):
                raise ValueError(f"ZIP entry escapes destination: {member!r}")
        zf.extractall(dest)


class SkillBundleStore:
    """Persists skill ZIP bundles in a document store and materializes them locally.

    Args:
        document_store: The system document store (LocalFS or S3) holding bundles.
        cache_dir:      Local directory where bundles are extracted for execution.
    """

    def __init__(self, document_store: DocumentStoreBase, cache_dir: str | Path = _SKILLS_CACHE_DIR) -> None:
        self._store = document_store
        self._cache_dir = Path(cache_dir)

    def skill_dir(self, skill_id: str) -> Path:
        return self._cache_dir / skill_id

    async def save_bundle(self, skill_id: str, zip_bytes: bytes) -> str:
        """Persist *zip_bytes* to the document store; return the bundle key."""
        key = bundle_key(skill_id)
        await self._store.save_bytes(SKILLS_COLLECTION, key, zip_bytes)
        return key

    def materialize(self, skill_id: str, zip_bytes: bytes) -> Path:
        """Extract *zip_bytes* into the local cache; return the dir holding SKILL.md.

        Raises ``ValueError`` if the bundle is unsafe or contains no ``SKILL.md``.
        """
        target = self.skill_dir(skill_id)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        try:
            _safe_extract(zip_bytes, target)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        skill_root = _find_skill_md(target)
        if skill_root is None:
            shutil.rmtree(target, ignore_errors=True)
            raise ValueError("bundle does not contain a SKILL.md")
        return skill_root

    async def sync_from_store(self, skill_id: str) -> Path:
        """Ensure the skill is materialized locally, fetching the bundle if needed.

        Returns the dir holding SKILL.md. Used on cold start / a fresh node.
        """
        existing = self.skill_dir(skill_id)
        if existing.exists():
            found = _find_skill_md(existing)
            if found is not None:
                return found
        zip_bytes = await self._store.load_bytes(SKILLS_COLLECTION, bundle_key(skill_id))
        return self.materialize(skill_id, zip_bytes)

    async def delete(self, skill_id: str) -> None:
        """Remove the bundle from the document store and the local cache."""
        try:
            await self._store.delete(SKILLS_COLLECTION, bundle_key(skill_id))
        except Exception as exc:  # best-effort; local cleanup still proceeds
            logger.warning("[skills] failed to delete bundle for %s: %s", skill_id, exc)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: shutil.rmtree(self.skill_dir(skill_id), ignore_errors=True))
