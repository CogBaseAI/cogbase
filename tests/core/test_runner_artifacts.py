"""Unit tests for the QueryRunner artifact primitives.

Covers the two general file-transport tools added for file-producing skills:

  fetch_document — materialize a stored original file to a local path
  save_artifact  — persist a skill-produced file for later download

plus their exposure gating (base tools, only offered when a skill is active).

A real ``LocalFSDocumentStore`` backs the tests so ``save_bytes`` / ``load_bytes``
round-trip on disk exactly as they do in production.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from cogbase.core.query_runner import MemoryTiers, QueryRunner, RetrievalResources
from cogbase.stores.document.local_fs import LocalFSDocumentStore


def _runner(store, app_id="app1") -> QueryRunner:
    return QueryRunner(
        app_id,
        MagicMock(),
        RetrievalResources(document_store=store),
        MemoryTiers(),
    )


# ---------------------------------------------------------------------------
# fetch_document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_document_materializes_original_to_local_path(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    await store.save_bytes("app1", "originals/contract1.docx", b"PK-docx-bytes")
    runner = _runner(store)

    out = await runner._run_fetch_document({"doc_id": "contract1"})

    assert "Fetched document 'contract1'" in out
    path = out.rsplit(" to ", 1)[1]
    assert os.path.exists(path)
    assert path.endswith(".docx")
    with open(path, "rb") as f:
        assert f.read() == b"PK-docx-bytes"


@pytest.mark.asyncio
async def test_fetch_document_falls_back_to_suffixless_key(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    # No .docx original — only a suffix-free key exists.
    await store.save_bytes("app1", "originals/note", b"raw")
    runner = _runner(store)

    out = await runner._run_fetch_document({"doc_id": "note"})

    assert "Fetched document 'note'" in out
    path = out.rsplit(" to ", 1)[1]
    with open(path, "rb") as f:
        assert f.read() == b"raw"


@pytest.mark.asyncio
async def test_fetch_document_missing_returns_error(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    out = await runner._run_fetch_document({"doc_id": "ghost"})
    assert out == "fetch_document error: no original file for 'ghost'"


@pytest.mark.asyncio
async def test_fetch_document_requires_doc_id(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    out = await runner._run_fetch_document({})
    assert out == "fetch_document error: doc_id is required"


# ---------------------------------------------------------------------------
# save_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_artifact_persists_under_generated_and_returns_id(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    runner = _runner(store)
    src = tmp_path / "merged.docx"
    src.write_bytes(b"merged-content")

    out = await runner._run_save_artifact({"path": str(src), "filename": "My Contract.docx"})

    assert "Saved artifact" in out
    artifact_id = out.split("'")[1]
    # Filename is sanitized, keeps its extension, and carries a uuid suffix.
    assert artifact_id.startswith("My_Contract__")
    assert artifact_id.endswith(".docx")
    stored = await store.load_bytes("app1", f"generated/{artifact_id}")
    assert stored == b"merged-content"


@pytest.mark.asyncio
async def test_save_artifact_defaults_filename_to_basename(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    runner = _runner(store)
    src = tmp_path / "out.docx"
    src.write_bytes(b"x")

    out = await runner._run_save_artifact({"path": str(src)})

    artifact_id = out.split("'")[1]
    assert artifact_id.startswith("out__")
    assert artifact_id.endswith(".docx")


@pytest.mark.asyncio
async def test_save_artifact_ids_are_unique_per_call(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    runner = _runner(store)
    src = tmp_path / "out.docx"
    src.write_bytes(b"x")

    first = (await runner._run_save_artifact({"path": str(src)})).split("'")[1]
    second = (await runner._run_save_artifact({"path": str(src)})).split("'")[1]
    assert first != second


@pytest.mark.asyncio
async def test_save_artifact_missing_file_returns_error(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    out = await runner._run_save_artifact({"path": "/no/such/file.docx"})
    assert out.startswith("save_artifact error: file not found")


@pytest.mark.asyncio
async def test_save_artifact_store_without_binary_support(tmp_path):
    """A store without save_bytes surfaces a clear message, not a raw exception."""
    from cogbase.stores.document.memory import InMemoryDocumentStore

    runner = _runner(InMemoryDocumentStore())
    src = tmp_path / "out.docx"
    src.write_bytes(b"x")

    out = await runner._run_save_artifact({"path": str(src)})
    assert "does not support binary artifacts" in out


# ---------------------------------------------------------------------------
# round-trip + tool gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_then_save_round_trip(tmp_path):
    """A skill's typical flow: fetch an original, then save a produced artifact."""
    store = LocalFSDocumentStore(str(tmp_path))
    await store.save_bytes("app1", "originals/c.docx", b"orig")
    runner = _runner(store)

    fetched_path = (await runner._run_fetch_document({"doc_id": "c"})).rsplit(" to ", 1)[1]
    # Simulate a skill editing the fetched file and writing an output.
    produced = tmp_path / "produced.docx"
    produced.write_bytes(open(fetched_path, "rb").read() + b"+edits")

    out = await runner._run_save_artifact({"path": str(produced), "filename": "c-merged.docx"})
    artifact_id = out.split("'")[1]
    assert await store.load_bytes("app1", f"generated/{artifact_id}") == b"orig+edits"


def test_artifact_tools_only_exposed_when_skill_active(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))

    active = {t["name"] for t in runner._all_tools(skill_active=True)}
    inactive = {t["name"] for t in runner._all_tools(skill_active=False)}

    assert {"fetch_document", "save_artifact"} <= active
    assert not ({"fetch_document", "save_artifact"} & inactive)
