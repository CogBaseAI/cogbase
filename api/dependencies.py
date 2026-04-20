"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from api.registry import AppRegistry
from api.system_store import SystemStore


def get_system_store(request: Request) -> SystemStore:
    return request.app.state.system_store  # type: ignore[no-any-return]


def get_registry(request: Request) -> AppRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


SystemStoreDep = Annotated[SystemStore, Depends(get_system_store)]
RegistryDep = Annotated[AppRegistry, Depends(get_registry)]
