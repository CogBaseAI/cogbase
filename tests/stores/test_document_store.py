"""Tests for document stores."""

import pytest

from cogbase.stores import DocumentStoreBase, LocalFSDocumentStore


def test_document_store_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DocumentStoreBase()  # type: ignore[abstract]


async def test_local_fs_save_load_exists_delete_roundtrip(tmp_path):
    store = LocalFSDocumentStore(tmp_path)
    doc_id = "doc-1.txt"
    content = "hello document store"

    assert await store.exists(doc_id) is False

    await store.save(doc_id, content)
    assert await store.exists(doc_id) is True
    assert await store.load(doc_id) == content

    await store.delete(doc_id)
    assert await store.exists(doc_id) is False
    with pytest.raises(KeyError, match=doc_id):
        await store.load(doc_id)


async def test_local_fs_overwrite_existing_document(tmp_path):
    store = LocalFSDocumentStore(tmp_path)
    doc_id = "doc-2"

    await store.save(doc_id, "v1")
    await store.save(doc_id, "v2")

    assert await store.load(doc_id) == "v2"


async def test_local_fs_supports_hierarchical_doc_ids(tmp_path):
    store = LocalFSDocumentStore(tmp_path)
    doc_id = "2026/q2/contracts/msa-1.txt"
    content = "nested path payload"

    await store.save(doc_id, content)
    assert await store.exists(doc_id) is True
    assert await store.load(doc_id) == content


async def test_local_fs_delete_missing_document_is_noop(tmp_path):
    store = LocalFSDocumentStore(tmp_path)
    await store.delete("missing-doc")


async def test_local_fs_rejects_path_escape(tmp_path):
    store = LocalFSDocumentStore(tmp_path)
    with pytest.raises(ValueError, match="escapes the store root"):
        await store.save("../outside.txt", "bad")
