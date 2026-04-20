"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from api.registry import AppRegistry
from api.system_config import SystemConfig
from api.system_store import SystemStore


def get_system_store(request: Request) -> SystemStore:
    return request.app.state.system_store  # type: ignore[no-any-return]


def get_registry(request: Request) -> AppRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


def get_system_config(request: Request) -> SystemConfig:
    return request.app.state.system_config  # type: ignore[no-any-return]


def get_system_structured_store(request: Request) -> Any:
    return request.app.state.system_structured_store


SystemStoreDep = Annotated[SystemStore, Depends(get_system_store)]
RegistryDep = Annotated[AppRegistry, Depends(get_registry)]
SystemConfigDep = Annotated[SystemConfig, Depends(get_system_config)]
SystemStructuredStoreDep = Annotated[Any, Depends(get_system_structured_store)]
