"""Unit tests for cogbase/skills/store.py — SkillBundleStore."""

from __future__ import annotations

import io
import zipfile

import pytest

from cogbase.skills.store import SkillBundleStore
from cogbase.stores.document.local_fs import LocalFSDocumentStore


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


VALID_MD = "---\nname: greeter\ndescription: Says hi.\n---\n# Greeter\nRun `python hello.py`.\n"


@pytest.fixture
def bundle_store(tmp_path):
    doc_store = LocalFSDocumentStore(tmp_path / "docs")
    return SkillBundleStore(doc_store, cache_dir=tmp_path / "cache")


def test_materialize_extracts_and_finds_skill_md(bundle_store):
    raw = _zip({"SKILL.md": VALID_MD, "hello.py": "print('hi')\n"})
    root = bundle_store.materialize("sk1", raw)
    assert (root / "SKILL.md").read_text() == VALID_MD
    assert (root / "hello.py").exists()


def test_materialize_finds_nested_skill_md(bundle_store):
    raw = _zip({"greeter/SKILL.md": VALID_MD, "greeter/hello.py": "print('hi')\n"})
    root = bundle_store.materialize("sk2", raw)
    assert (root / "SKILL.md").exists()
    assert root.name == "greeter"


def test_materialize_rejects_missing_skill_md(bundle_store):
    raw = _zip({"readme.txt": "no skill here"})
    with pytest.raises(ValueError, match="SKILL.md"):
        bundle_store.materialize("sk3", raw)
    assert not bundle_store.skill_dir("sk3").exists()


def test_materialize_rejects_zip_slip(bundle_store):
    raw = _zip({"../escape.txt": "pwned", "SKILL.md": VALID_MD})
    with pytest.raises(ValueError, match="escapes destination"):
        bundle_store.materialize("sk4", raw)


def test_materialize_overwrites_existing(bundle_store):
    bundle_store.materialize("sk5", _zip({"SKILL.md": VALID_MD, "old.py": "1"}))
    root = bundle_store.materialize("sk5", _zip({"SKILL.md": VALID_MD, "new.py": "2"}))
    assert (root / "new.py").exists()
    assert not (root / "old.py").exists()


@pytest.mark.asyncio
async def test_sync_from_store_round_trip(bundle_store):
    raw = _zip({"SKILL.md": VALID_MD, "hello.py": "print('hi')\n"})
    await bundle_store.save_bundle("sk6", raw)

    # Simulate a fresh node: drop the local cache, then sync from the doc store.
    import shutil
    shutil.rmtree(bundle_store.skill_dir("sk6"), ignore_errors=True)
    assert not bundle_store.skill_dir("sk6").exists()

    root = await bundle_store.sync_from_store("sk6")
    assert (root / "SKILL.md").read_text() == VALID_MD


@pytest.mark.asyncio
async def test_delete_removes_bundle_and_cache(bundle_store):
    raw = _zip({"SKILL.md": VALID_MD})
    await bundle_store.save_bundle("sk7", raw)
    bundle_store.materialize("sk7", raw)
    assert bundle_store.skill_dir("sk7").exists()

    await bundle_store.delete("sk7")
    assert not bundle_store.skill_dir("sk7").exists()
    with pytest.raises(KeyError):
        await bundle_store.sync_from_store("sk7")
