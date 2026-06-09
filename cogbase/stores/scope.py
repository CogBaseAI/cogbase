"""Scope descriptor for isolating store collections by account / namespace / app.

All three levels are *stable internal ids*, not client-facing names — analogous
to database and table. Clients address an application by its (mutable) name;
the scope is keyed by ids so renaming never moves the underlying storage.
Only ``app_id`` is populated today; ``account_id`` / ``namespace_id`` are wired
when tenancy/auth lands.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppScope:
    """Identifies where a store's collections live in the id hierarchy.

    Fields are all optional ids; only the non-None ones contribute to the prefix.
    The separator between parts defaults to ``__`` (double-underscore).

    Examples::

        AppScope(app_id="a1b2").prefix()                            # "a1b2"
        AppScope(namespace_id="eng", app_id="a1b2").prefix()        # "eng__a1b2"
        AppScope(account_id="acme", namespace_id="eng", app_id="a1b2").prefix()
        # "acme__eng__a1b2"
    """

    account_id: str | None = None
    namespace_id: str | None = None
    app_id: str | None = None

    def prefix(self, sep: str = "__") -> str | None:
        parts = [p for p in (self.account_id, self.namespace_id, self.app_id) if p]
        return sep.join(parts) if parts else None
