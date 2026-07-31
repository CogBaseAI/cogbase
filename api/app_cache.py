"""LRU cache of live application instances."""

from __future__ import annotations

from typing import Any

from cachetools import TTLCache
from threading import Lock


def cache_key(account_id: str, namespace_id: str, name: str) -> str:
    """Composite cache key for a live app instance.

    An app's client-facing ``name`` is only unique within ``(account, namespace)``,
    so the cache is keyed by the full tuple — two tenants (or two namespaces of one
    account) may hold apps of the same name without colliding.
    """
    return f"{account_id}/{namespace_id}/{name}"


class AppCache:
    """Maps ``account/namespace/name`` → live pack application instance, LRU-backed.

    Populated at startup from the system store and kept warm by CRUD endpoints.
    On a cache miss the caller is responsible for rebuilding from the system store.
    Keys are built with :func:`cache_key`.
    """

    def __init__(self, maxsize: int = 256, ttl: int = 60) -> None:
        self._cache = TTLCache(maxsize, ttl)
        self.lock = Lock()

    def add(self, key: str, app: Any) -> None:
        with self.lock:
            self._cache[key] = app

    def get(self, key: str) -> Any | None:
        with self.lock:
            return self._cache.get(key)

    def remove(self, key: str) -> None:
        with self.lock:
            self._cache.pop(key, None)

    def apps_for_account(self, account_id: str) -> list[Any]:
        """Return every live instance the cache holds for one account.

        Keys are ``account/namespace/name`` (:func:`cache_key`), so an exact
        prefix match selects an account's apps across all its namespaces. Used to
        push an account-scoped change (e.g. an edited company profile) into
        already-built instances instead of dropping them and paying a rebuild.

        Snapshots under the lock and reads through ``get`` so an entry expiring
        mid-iteration is skipped rather than raising.
        """
        prefix = f"{account_id}/"
        with self.lock:
            keys = [k for k in list(self._cache.keys()) if k.startswith(prefix)]
            apps = [self._cache.get(k) for k in keys]
        return [app for app in apps if app is not None]

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()
