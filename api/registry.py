"""In-memory registry of live application instances."""

from __future__ import annotations

from typing import Any


class AppRegistry:
    """Maps app_id → live pack application instance.

    Populated at startup from the system store and updated as applications are
    created, updated, or deleted via the REST API.
    """

    def __init__(self) -> None:
        self._apps: dict[str, Any] = {}

    def add(self, app_id: str, app: Any) -> None:
        self._apps[app_id] = app

    def get(self, app_id: str) -> Any | None:
        return self._apps.get(app_id)

    def remove(self, app_id: str) -> None:
        self._apps.pop(app_id, None)

    def all_ids(self) -> list[str]:
        return list(self._apps.keys())
