"""LRU cache of live application instances."""

from __future__ import annotations

from typing import Any

from cachetools import TTLCache
from threading import Lock


class AppCache:
    """Maps app name → live pack application instance, backed by an LRU cache.

    Populated at startup from the system store and kept warm by CRUD endpoints.
    On a cache miss the caller is responsible for rebuilding from the system store.
    """

    def __init__(self, maxsize: int = 256, ttl: int = 60) -> None:
        self._cache = TTLCache(maxsize, ttl)
        self.lock = Lock()

    def add(self, name: str, app: Any) -> None:
        with self.lock:
            self._cache[name] = app

    def get(self, name: str) -> Any | None:
        with self.lock:
            return self._cache.get(name)

    def remove(self, name: str) -> None:
        with self.lock:
            self._cache.pop(name, None)
