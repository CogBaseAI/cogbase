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

from cogbase.core.query_runner import (
    ArtifactRef,
    MemoryTiers,
    QueryRunner,
    RetrievalResources,
    _append_download_links,
)
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

    artifact, out = await runner._run_save_artifact({"path": str(src), "filename": "My Contract.docx"})

    assert "Saved artifact" in out
    assert artifact is not None
    artifact_id = artifact.artifact_id
    # Filename is sanitized, keeps its extension, and carries a uuid suffix.
    assert artifact_id.startswith("My_Contract__")
    assert artifact_id.endswith(".docx")
    stored = await store.load_bytes("app1", f"generated/{artifact_id}")
    assert stored == b"merged-content"


@pytest.mark.asyncio
async def test_save_artifact_returns_ready_markdown_download_link(tmp_path):
    """The tool output and ArtifactRef carry an app_id-scoped markdown download link."""
    store = LocalFSDocumentStore(str(tmp_path))
    runner = _runner(store, app_id="app-123")
    src = tmp_path / "merged.docx"
    src.write_bytes(b"x")

    artifact, out = await runner._run_save_artifact({"path": str(src), "filename": "final.docx"})

    # Keyed by the stable app_id so the link survives a rename.
    expected_path = f"/applications/app-123/documents/{artifact.artifact_id}/download"
    assert artifact.download_path == expected_path
    assert artifact.markdown_link == f"[final.docx]({expected_path})"
    # The model is handed the exact link to reproduce.
    assert artifact.markdown_link in out


@pytest.mark.asyncio
async def test_save_artifact_defaults_filename_to_basename(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    runner = _runner(store)
    src = tmp_path / "out.docx"
    src.write_bytes(b"x")

    artifact, _ = await runner._run_save_artifact({"path": str(src)})

    assert artifact.artifact_id.startswith("out__")
    assert artifact.artifact_id.endswith(".docx")
    assert artifact.filename == "out.docx"


@pytest.mark.asyncio
async def test_save_artifact_ids_are_unique_per_call(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    runner = _runner(store)
    src = tmp_path / "out.docx"
    src.write_bytes(b"x")

    first, _ = await runner._run_save_artifact({"path": str(src)})
    second, _ = await runner._run_save_artifact({"path": str(src)})
    assert first.artifact_id != second.artifact_id


@pytest.mark.asyncio
async def test_save_artifact_missing_file_returns_error(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    artifact, out = await runner._run_save_artifact({"path": "/no/such/file.docx"})
    assert artifact is None
    assert out.startswith("save_artifact error: file not found")


@pytest.mark.asyncio
async def test_save_artifact_store_without_binary_support(tmp_path):
    """A store without save_bytes surfaces a clear message, not a raw exception."""
    from cogbase.stores.document.memory import InMemoryDocumentStore

    runner = _runner(InMemoryDocumentStore())
    src = tmp_path / "out.docx"
    src.write_bytes(b"x")

    artifact, out = await runner._run_save_artifact({"path": str(src)})
    assert artifact is None
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

    artifact, _ = await runner._run_save_artifact({"path": str(produced), "filename": "c-merged.docx"})
    assert await store.load_bytes("app1", f"generated/{artifact.artifact_id}") == b"orig+edits"


# ---------------------------------------------------------------------------
# download-link appending
# ---------------------------------------------------------------------------


def _ref(filename="out.docx", artifact_id="out__abc123.docx") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        filename=filename,
        download_path=f"/applications/app/documents/{artifact_id}/download",
    )


def test_append_download_links_adds_markdown_block_for_missing_artifact():
    ref = _ref()
    out = _append_download_links("Here is your revised document.\n", [ref])
    assert ref.markdown_link in out
    assert "**Download:**" in out


def test_append_download_links_skips_artifact_already_linked():
    ref = _ref()
    # Model already wrote the exact link; no duplicate block is appended.
    answer = f"Done — {ref.markdown_link}\n"
    assert _append_download_links(answer, [ref]) == answer


def test_append_download_links_noop_without_artifacts():
    assert _append_download_links("answer\n", []) == "answer\n"


def test_artifact_tools_only_exposed_when_skill_active(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))

    active = {t["name"] for t in runner._all_tools(skill_active=True)}
    inactive = {t["name"] for t in runner._all_tools(skill_active=False)}

    assert {"fetch_document", "save_artifact"} <= active
    assert not ({"fetch_document", "save_artifact"} & inactive)
