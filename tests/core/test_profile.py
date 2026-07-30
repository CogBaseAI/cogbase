"""Tests for the account-scoped company profile store."""

from __future__ import annotations

import pytest

from cogbase.core.profile import (
    COMPANY_PROFILE_DOC_ID,
    MAX_PROFILE_BYTES,
    PROFILE_COLLECTION,
    AccountProfileStore,
)
from cogbase.stores.document.memory import InMemoryDocumentStore
from cogbase.stores.scope import AppScope

PROFILE = "# Company Profile\n\n## Who we are\nWe sell widgets.\n"


async def test_save_load_round_trip():
    store = AccountProfileStore(InMemoryDocumentStore())

    await store.save("acme", PROFILE)

    assert await store.load("acme") == PROFILE
    assert await store.exists("acme") is True


async def test_load_missing_profile_returns_none():
    """Absence is the cold-start state, not an error."""
    store = AccountProfileStore(InMemoryDocumentStore())

    assert await store.load("acme") is None
    assert await store.exists("acme") is False


async def test_save_overwrites_previous_version():
    store = AccountProfileStore(InMemoryDocumentStore())

    await store.save("acme", PROFILE)
    await store.save("acme", "# Company Profile\n\nRevised.\n")

    assert await store.load("acme") == "# Company Profile\n\nRevised.\n"


async def test_delete_removes_profile_and_is_idempotent():
    store = AccountProfileStore(InMemoryDocumentStore())
    await store.save("acme", PROFILE)

    await store.delete("acme")
    assert await store.load("acme") is None

    await store.delete("acme")  # no-op, must not raise


async def test_accounts_are_isolated():
    store = AccountProfileStore(InMemoryDocumentStore())

    await store.save("acme", PROFILE)
    await store.save("globex", "# Company Profile\n\nWe sell sprockets.\n")

    assert await store.load("acme") == PROFILE
    assert "sprockets" in await store.load("globex")
    await store.delete("acme")
    assert await store.load("globex") is not None


async def test_profile_key_is_account_only_so_every_namespace_and_app_shares_it():
    """One document per account: the scope carries no namespace or app id.

    Asserted on the backing key layout, since that layout *is* the sharing
    mechanism — an app-scoped view of the same store prefixes its collections with
    ``account__namespace__app`` and so could never reach this document.
    """
    backing = InMemoryDocumentStore()
    store = AccountProfileStore(backing)

    await store.save("acme", PROFILE)

    assert list(backing._store) == [(f"acme__{PROFILE_COLLECTION}", COMPANY_PROFILE_DOC_ID)]

    app_scoped = backing.with_scope(
        AppScope(account_id="acme", namespace_id="legal", app_id="a1b2")
    )
    assert await app_scoped.exists(PROFILE_COLLECTION, COMPANY_PROFILE_DOC_ID) is False


async def test_save_rejects_oversized_profile():
    store = AccountProfileStore(InMemoryDocumentStore())

    with pytest.raises(ValueError, match="over the"):
        await store.save("acme", "x" * (MAX_PROFILE_BYTES + 1))

    assert await store.load("acme") is None


async def test_size_limit_counts_utf8_bytes_not_characters():
    store = AccountProfileStore(InMemoryDocumentStore())

    # Multi-byte characters: comfortably under the limit by character count,
    # over it by encoded size.
    with pytest.raises(ValueError):
        await store.save("acme", "€" * (MAX_PROFILE_BYTES // 2))


async def test_load_propagates_store_failures():
    """Only a missing document is swallowed; a broken store must surface."""

    class BrokenStore(InMemoryDocumentStore):
        async def load(self, collection: str, doc_id: str) -> str:
            raise RuntimeError("document store unreachable")

    store = AccountProfileStore(BrokenStore())

    with pytest.raises(RuntimeError, match="unreachable"):
        await store.load("acme")
