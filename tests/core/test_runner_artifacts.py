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
# fetch_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_artifact_materializes_saved_artifact_to_local_path(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    await store.save_bytes("app1", "generated/ops__abc123.json", b'{"clauses": []}')
    runner = _runner(store)

    out = await runner._run_fetch_artifact({"artifact_id": "ops__abc123.json"})

    assert "Fetched artifact 'ops__abc123.json'" in out
    path = out.rsplit(" to ", 1)[1]
    assert os.path.exists(path)
    assert path.endswith(".json")
    with open(path, "rb") as f:
        assert f.read() == b'{"clauses": []}'


@pytest.mark.asyncio
async def test_fetch_artifact_missing_returns_error(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    out = await runner._run_fetch_artifact({"artifact_id": "ghost.json"})
    assert out == "fetch_artifact error: no artifact 'ghost.json'"


@pytest.mark.asyncio
async def test_fetch_artifact_requires_artifact_id(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    out = await runner._run_fetch_artifact({})
    assert out == "fetch_artifact error: artifact_id is required"


# ---------------------------------------------------------------------------
# delete_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_artifact_removes_stored_file(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path))
    await store.save_bytes("app1", "generated/ops__abc123.json", b"{}")
    runner = _runner(store)

    out = await runner._run_delete_artifact({"artifact_id": "ops__abc123.json"})

    assert out == "Deleted artifact 'ops__abc123.json'."
    with pytest.raises(KeyError):
        await store.load_bytes("app1", "generated/ops__abc123.json")


@pytest.mark.asyncio
async def test_delete_artifact_is_idempotent(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    out = await runner._run_delete_artifact({"artifact_id": "ghost.json"})
    assert out == "Deleted artifact 'ghost.json'."


@pytest.mark.asyncio
async def test_delete_artifact_requires_artifact_id(tmp_path):
    runner = _runner(LocalFSDocumentStore(str(tmp_path)))
    out = await runner._run_delete_artifact({})
    assert out == "delete_artifact error: artifact_id is required"


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


@pytest.mark.asyncio
async def test_save_fetch_patch_resave_round_trip(tmp_path):
    """The working-state loop: save ops.json, reload it, patch, save again."""
    store = LocalFSDocumentStore(str(tmp_path))
    runner = _runner(store)
    ops = tmp_path / "ops.json"
    ops.write_bytes(b'{"verdict": "pending"}')

    saved, _ = await runner._run_save_artifact({"path": str(ops), "filename": "ops.json"})

    fetched_path = (
        await runner._run_fetch_artifact({"artifact_id": saved.artifact_id})
    ).rsplit(" to ", 1)[1]
    with open(fetched_path, "rb") as f:
        assert f.read() == b'{"verdict": "pending"}'

    # Patch the reloaded working state and persist a fresh copy.
    patched = tmp_path / "ops2.json"
    patched.write_bytes(b'{"verdict": "accepted"}')
    resaved, _ = await runner._run_save_artifact({"path": str(patched), "filename": "ops.json"})
    assert await store.load_bytes("app1", f"generated/{resaved.artifact_id}") == b'{"verdict": "accepted"}'


# ---------------------------------------------------------------------------
# session working directory
# ---------------------------------------------------------------------------


def test_session_workdir_is_deterministic_per_session(tmp_path):
    runner = _runner(MagicMock(), app_id="app1")
    runner._work_root = str(tmp_path)

    first = runner._session_workdir("sess-A")
    again = runner._session_workdir("sess-A")
    other = runner._session_workdir("sess-B")

    assert first == again  # same (app, session) → same dir, so paths are predictable
    assert first != other
    assert os.path.isdir(first)
    assert first.startswith(os.path.join(str(tmp_path), "app1"))


def test_session_workdir_without_session_is_isolated(tmp_path):
    runner = _runner(MagicMock(), app_id="app1")
    runner._work_root = str(tmp_path)

    a = runner._session_workdir(None)
    b = runner._session_workdir(None)
    assert a != b  # no session → random bucket, keeps a turn's files together only


@pytest.mark.asyncio
async def test_fetch_document_writes_into_workdir_deterministically(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path / "store"))
    await store.save_bytes("app1", "originals/contract1.docx", b"PK-docx-bytes")
    runner = _runner(store)
    workdir = str(tmp_path / "work")

    out = await runner._run_fetch_document({"doc_id": "contract1"}, workdir)
    path = out.rsplit(" to ", 1)[1]

    assert path == os.path.join(workdir, "originals", "contract1.docx")
    with open(path, "rb") as f:
        assert f.read() == b"PK-docx-bytes"
    # A second fetch lands at the same path (no random suffix to lose track of).
    out2 = await runner._run_fetch_document({"doc_id": "contract1"}, workdir)
    assert out2.rsplit(" to ", 1)[1] == path


@pytest.mark.asyncio
async def test_fetch_artifact_writes_into_workdir(tmp_path):
    store = LocalFSDocumentStore(str(tmp_path / "store"))
    await store.save_bytes("app1", "generated/ops__abc123.json", b'{"clauses": []}')
    runner = _runner(store)
    workdir = str(tmp_path / "work")

    out = await runner._run_fetch_artifact({"artifact_id": "ops__abc123.json"}, workdir)
    path = out.rsplit(" to ", 1)[1]

    assert path == os.path.join(workdir, "ops__abc123.json")
    with open(path, "rb") as f:
        assert f.read() == b'{"clauses": []}'


def test_tool_env_exposes_workdir_and_skill_dir(tmp_path):
    from pathlib import Path

    runner = _runner(MagicMock())
    skill = MagicMock()
    skill.site_packages = None
    skill.source_path = Path("/skills/abc123/SKILL.md")

    env = runner._tool_env(skill, "/some/workdir")
    assert env["COGBASE_WORKDIR"] == "/some/workdir"
    # The skill dir is the SKILL.md's parent, so bundled scripts resolve reliably.
    assert env["COGBASE_SKILL_DIR"] == os.path.join(os.sep, "skills", "abc123")

    # Both omitted when their source is absent.
    bare = runner._tool_env(None, None)
    assert "COGBASE_WORKDIR" not in bare
    assert "COGBASE_SKILL_DIR" not in bare


def test_system_prompt_advertises_paths(tmp_path):
    from pathlib import Path

    runner = _runner(MagicMock())
    skill = MagicMock()
    skill.name = "legal-review"
    skill.source_path = Path("/skills/abc123/SKILL.md")
    skill.metadata = {}
    skill.raw_markdown = "body"

    prompt = runner.build_system_prompt("base", skill, "/work/app1/sess")
    assert "/work/app1/sess" in prompt and "COGBASE_WORKDIR" in prompt
    assert "/skills/abc123" in prompt and "COGBASE_SKILL_DIR" in prompt

    # No workdir → no workdir pointer, but the skill dir still shows.
    no_wd = runner.build_system_prompt("base", skill, None)
    assert "COGBASE_WORKDIR" not in no_wd and "COGBASE_SKILL_DIR" in no_wd


def _skill_for_prompt(tmp_path):
    from pathlib import Path

    skill = MagicMock()
    skill.name = "legal-review"
    skill.source_path = Path("/skills/abc123/SKILL.md")
    skill.metadata = {}
    skill.raw_markdown = "body"
    return skill


def test_system_prompt_lists_persisted_workdir_files(tmp_path):
    # A follow-up turn inherits the prior turn's materialized state; the prompt
    # advertises it so the model reuses review.json / the fetched original in
    # place rather than re-fetching and re-reviewing from scratch.
    runner = _runner(MagicMock())
    workdir = tmp_path / "work"
    (workdir / "originals").mkdir(parents=True)
    (workdir / "originals" / "base.docx").write_bytes(b"x" * 10)
    (workdir / "review.json").write_text("{}")
    (workdir / "redline.docx").write_bytes(b"y" * 20)

    prompt = runner.build_system_prompt("base", _skill_for_prompt(tmp_path), str(workdir))

    assert "Files already in the working directory" in prompt
    assert "`review.json` (2 bytes)" in prompt
    assert "`redline.docx` (20 bytes)" in prompt
    assert "`originals/base.docx` (10 bytes)" in prompt  # nested files shown by relative path


def test_system_prompt_omits_listing_for_empty_or_missing_workdir(tmp_path):
    runner = _runner(MagicMock())
    skill = _skill_for_prompt(tmp_path)

    # Non-existent workdir (fresh node) → still advertises the path, no listing.
    missing = runner.build_system_prompt("base", skill, str(tmp_path / "nope"))
    assert "COGBASE_WORKDIR" in missing
    assert "Files already in the working directory" not in missing

    # Existing but empty workdir → no listing block.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "Files already in the working directory" not in runner.build_system_prompt(
        "base", skill, str(empty)
    )


def test_system_prompt_caps_workdir_listing(tmp_path):
    runner = _runner(MagicMock())
    workdir = tmp_path / "work"
    workdir.mkdir()
    for i in range(60):
        (workdir / f"f{i:02d}.txt").write_text("x")

    prompt = runner.build_system_prompt("base", _skill_for_prompt(tmp_path), str(workdir))
    block = prompt.split("Files already in the working directory", 1)[1]
    # 50 entries + one "…and N more" summary line, not all 60.
    assert "…and 10 more" in block
    assert "`f00.txt`" in block and "`f49.txt`" in block
    assert "`f59.txt`" not in block


@pytest.mark.asyncio
async def test_shell_refuses_whole_filesystem_scan(tmp_path):
    runner = _runner(MagicMock())
    out = await runner._run_shell('find / -name "segment_clauses.py"', env={})
    assert "Refusing to scan the whole filesystem" in out
    assert "$COGBASE_SKILL_DIR" in out


@pytest.mark.asyncio
async def test_shell_allows_scoped_commands(tmp_path):
    runner = _runner(MagicMock())
    # A scoped path and an ordinary command are not filesystem-root scans.
    assert (await runner._run_shell("echo ok", env=dict(os.environ))) == "ok"
    from cogbase.core.query_runner import _ROOT_FS_SCAN
    assert _ROOT_FS_SCAN.search("find / -name x")          # root scan → blocked
    assert _ROOT_FS_SCAN.search("grep -r foo /")           # root scan → blocked
    assert not _ROOT_FS_SCAN.search("find /skills/abc -name x")  # scoped → allowed
    assert not _ROOT_FS_SCAN.search('find "$COGBASE_WORKDIR" -type f')  # anchored → allowed
    assert not _ROOT_FS_SCAN.search("find . -name x")      # relative → allowed


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

    artifact_tools = {"fetch_document", "save_artifact", "fetch_artifact", "delete_artifact"}
    assert artifact_tools <= active
    assert not (artifact_tools & inactive)
