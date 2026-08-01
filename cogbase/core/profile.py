"""Account profile — the account-scoped company profile document.

The company profile is stable org-wide context a customer supplies once: who they
are, jurisdictions, regulators, risk appetite, house style, role.  None of it is
derivable from their documents, and none of it should be re-collected per app — so
it lives *outside* every app partition, as one markdown document per account (see
docs/preference-profiles.md).

The body is prose, not YAML: a document the user reads and edits.
The query runner is the consumer: ``api/factory.py`` reads the profile at app
build time and the runner injects it unconditionally as framing context.

Storage is the system document store, viewed through
``AppScope(account_id=account_id)``.  Dropping ``namespace_id`` and ``app_id`` from
the scope is what makes the document shared: only the non-None parts of a scope
contribute to the collection prefix (see ``cogbase.stores.scope``), so every
namespace and app in the account addresses the same collection.
"""

from __future__ import annotations

import logging

from cogbase.stores.document.base import DocumentStoreBase
from cogbase.stores.scope import AppScope

logger = logging.getLogger(__name__)

PROFILE_COLLECTION = "profile"
COMPANY_PROFILE_DOC_ID = "company-profile.md"

# A page of prose that rides in every system prompt for the account.  The cap is
# about prompt budget, not storage: past a certain size the profile stops being
# framing context and starts crowding out the documents the answer is grounded in.
MAX_PROFILE_BYTES = 32 * 1024


class AccountProfileStore:
    """Reads and writes one ``company-profile.md`` per account.

    Takes the *unscoped* system document store and applies the account scope per
    call, so a single instance serves every account.

    Example::

        store = AccountProfileStore(system_resources.document_store)
        await store.save("acme", "# Company Profile\\n...")
        profile = await store.load("acme")   # None when the account has no profile
    """

    def __init__(self, document_store: DocumentStoreBase) -> None:
        self._store = document_store

    def _scoped(self, account_id: str) -> DocumentStoreBase:
        return self._store.with_scope(AppScope(account_id=account_id))

    async def load(self, account_id: str) -> str | None:
        """Return the account's profile markdown, or ``None`` if it has none.

        Absence is the normal cold-start state — an account that has not been
        through the onboarding interview yet — so it is reported as ``None``
        rather than raised.  A store-level failure still propagates; callers on
        the request path decide whether to degrade (``build_app`` treats the read
        as best-effort so a doc-store hiccup cannot fail app construction).
        """
        try:
            return await self._scoped(account_id).load(PROFILE_COLLECTION, COMPANY_PROFILE_DOC_ID)
        except KeyError:
            return None

    async def save(self, account_id: str, markdown: str) -> None:
        """Persist *markdown* as the account's profile, replacing any previous version.

        Raises ``ValueError`` when the body exceeds ``MAX_PROFILE_BYTES``.  The
        check lives here, at the durable boundary, so both writers — the API's
        edit endpoint and the onboarding interview's tool — are held to it; the
        API translates the error into its own status code.
        """
        size = len(markdown.encode("utf-8"))
        if size > MAX_PROFILE_BYTES:
            raise ValueError(
                f"company profile is {size} bytes, over the {MAX_PROFILE_BYTES}-byte limit"
            )
        await self._scoped(account_id).save(
            PROFILE_COLLECTION, COMPANY_PROFILE_DOC_ID, markdown
        )
        logger.info("saved company profile account=%s bytes=%d", account_id, size)

    async def exists(self, account_id: str) -> bool:
        """Return ``True`` when the account has a profile."""
        return await self._scoped(account_id).exists(
            PROFILE_COLLECTION, COMPANY_PROFILE_DOC_ID
        )

    async def delete(self, account_id: str) -> None:
        """Delete the account's profile.  No-op when it has none."""
        await self._scoped(account_id).delete(PROFILE_COLLECTION, COMPANY_PROFILE_DOC_ID)
        logger.info("deleted company profile account=%s", account_id)
