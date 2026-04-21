"""Unit tests for api/cacheistry.py — AppCache."""

from __future__ import annotations

import pytest

from api.app_cache import AppCache


class TestAppCache:
    def _all_names(self, app_cache) -> list[str]:
        return list(app_cache._cache.keys())

    def test_empty_on_creation(self):
        cache = AppCache()
        assert self._all_names(cache) == []

    def test_add_and_get(self):
        cache = AppCache()
        app = object()
        cache.add("app-1", app)
        assert cache.get("app-1") is app

    def test_get_unknown_returns_none(self):
        cache = AppCache()
        assert cache.get("nonexistent") is None

    def test_add_multiple(self):
        cache = AppCache()
        a, b = object(), object()
        cache.add("app-1", a)
        cache.add("app-2", b)
        assert set(self._all_names(cache)) == {"app-1", "app-2"}
        assert cache.get("app-1") is a
        assert cache.get("app-2") is b

    def test_add_overwrites_existing(self):
        cache = AppCache()
        old, new = object(), object()
        cache.add("app-1", old)
        cache.add("app-1", new)
        assert cache.get("app-1") is new
        assert len(self._all_names(cache)) == 1

    def test_remove_known(self):
        cache = AppCache()
        cache.add("app-1", object())
        cache.remove("app-1")
        assert cache.get("app-1") is None
        assert "app-1" not in self._all_names(cache)

    def test_remove_unknown_is_noop(self):
        cache = AppCache()
        cache.remove("ghost")  # must not raise

    def test_all_ids_returns_list(self):
        cache = AppCache()
        cache.add("x", object())
        cache.add("y", object())
        ids = self._all_names(cache)
        assert isinstance(ids, list)
        assert set(ids) == {"x", "y"}

    def test_remove_does_not_affect_others(self):
        cache = AppCache()
        cache.add("app-1", object())
        cache.add("app-2", object())
        cache.remove("app-1")
        assert cache.get("app-2") is not None
        assert len(self._all_names(cache)) == 1

    def test_lru_evicts_oldest_when_full(self):
        cache = AppCache(maxsize=2)
        a, b, c = object(), object(), object()
        cache.add("a", a)
        cache.add("b", b)
        cache.add("c", c)  # "a" should be evicted
        assert cache.get("a") is None
        assert cache.get("b") is b
        assert cache.get("c") is c
