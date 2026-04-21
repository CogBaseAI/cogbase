"""In-memory registry of live application instances."""

from __future__ import annotations

from typing import Any


class AppRegistry:
    """Maps app name → live pack application instance.

    Populated at startup from the system store and updated as applications are
    created, updated, or deleted via the REST API.
    """

    def __init__(self) -> None:
        self._apps: dict[str, Any] = {}

    def add(self, name: str, app: Any) -> None:
        self._apps[name] = app

    def get(self, name: str) -> Any | None:
        return self._apps.get(name)

    def remove(self, name: str) -> None:
        self._apps.pop(name, None)

    def all_names(self) -> list[str]:
        return list(self._apps.keys())
