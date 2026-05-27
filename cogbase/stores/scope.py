"""Scope descriptor for isolating store collections by account / namespace / app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppScope:
    """Identifies where a store's collections live in the hierarchy.

    Fields are all optional; only the non-None ones contribute to the prefix.
    The separator between parts defaults to ``__`` (double-underscore).

    Examples::

        AppScope(app="myapp").prefix()                     # "myapp"
        AppScope(namespace="eng", app="myapp").prefix()   # "eng__myapp"
        AppScope(account="acme", namespace="eng", app="myapp").prefix()
        # "acme__eng__myapp"
    """

    account: str | None = None
    namespace: str | None = None
    app: str | None = None

    def prefix(self, sep: str = "__") -> str | None:
        parts = [p for p in (self.account, self.namespace, self.app) if p]
        return sep.join(parts) if parts else None
