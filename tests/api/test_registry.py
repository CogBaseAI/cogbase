"""Unit tests for api/registry.py — AppRegistry."""

from __future__ import annotations

import pytest

from api.registry import AppRegistry


class TestAppRegistry:
    def test_empty_on_creation(self):
        reg = AppRegistry()
        assert reg.all_names() == []

    def test_add_and_get(self):
        reg = AppRegistry()
        app = object()
        reg.add("app-1", app)
        assert reg.get("app-1") is app

    def test_get_unknown_returns_none(self):
        reg = AppRegistry()
        assert reg.get("nonexistent") is None

    def test_add_multiple(self):
        reg = AppRegistry()
        a, b = object(), object()
        reg.add("app-1", a)
        reg.add("app-2", b)
        assert set(reg.all_names()) == {"app-1", "app-2"}
        assert reg.get("app-1") is a
        assert reg.get("app-2") is b

    def test_add_overwrites_existing(self):
        reg = AppRegistry()
        old, new = object(), object()
        reg.add("app-1", old)
        reg.add("app-1", new)
        assert reg.get("app-1") is new
        assert len(reg.all_names()) == 1

    def test_remove_known(self):
        reg = AppRegistry()
        reg.add("app-1", object())
        reg.remove("app-1")
        assert reg.get("app-1") is None
        assert "app-1" not in reg.all_names()

    def test_remove_unknown_is_noop(self):
        reg = AppRegistry()
        reg.remove("ghost")  # must not raise

    def test_all_ids_returns_list(self):
        reg = AppRegistry()
        reg.add("x", object())
        reg.add("y", object())
        ids = reg.all_names()
        assert isinstance(ids, list)
        assert set(ids) == {"x", "y"}

    def test_remove_does_not_affect_others(self):
        reg = AppRegistry()
        reg.add("app-1", object())
        reg.add("app-2", object())
        reg.remove("app-1")
        assert reg.get("app-2") is not None
        assert len(reg.all_names()) == 1
