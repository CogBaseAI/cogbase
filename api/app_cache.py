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

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()
