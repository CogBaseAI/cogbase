"""FastAPI dependency providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request

from api.app_cache import AppCache
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.store import SkillBundleStore


#: Default account/namespace used when a request omits the tenancy header.
#: Tenancy is logical for now — ``account_id`` is trust-on-declaration until an
#: auth layer binds the header to an authenticated principal.
DEFAULT_ACCOUNT_ID = "default"
DEFAULT_NAMESPACE = "default"

#: How this instance resolves the calling account, set by the operator at deploy
#: time via ``COGBASE_DEPLOYMENT_MODE``. It is advisory metadata the UI reads from
#: ``GET /whoami`` to decide whether to expose an account switcher:
#:   - ``dev`` (default): account is trust-on-declaration via the X-Account-Id
#:     header, so the UI keeps an editable account field.
#:   - ``saas`` / ``single_tenant`` / ``demo``: the account is server-authoritative
#:     (derived from the host/session or fixed at deploy), so the UI treats the
#:     account returned by /whoami as read-only.
#: The value does not yet change server-side resolution — it is the seam that will,
#: once an auth layer binds the account to an authenticated principal.
DEPLOYMENT_MODE = os.environ.get("COGBASE_DEPLOYMENT_MODE", "dev")


def get_deployment_mode() -> str:
    """Return the operator-declared deployment mode (see :data:`DEPLOYMENT_MODE`)."""
    return DEPLOYMENT_MODE


def get_account_id(
    x_account_id: Annotated[str | None, Header()] = None,
) -> str:
    """Resolve the calling tenant from the ``X-Account-Id`` header.

    Falls back to ``DEFAULT_ACCOUNT_ID`` so single-tenant callers keep working.
    The account is the security boundary; the namespace is addressed in the path.
    """
    return x_account_id or DEFAULT_ACCOUNT_ID


def resolve_namespace_id(account_id: str, name: str) -> str:
    """Map a user-facing namespace ``name`` to its internal ``namespace_id``.

    A namespace is the first layer inside an account, so ``name`` is unique per
    account and addresses exactly one namespace. The record already stores the
    name and id as separate columns; today they coincide (the name is minted as
    the id at creation), so this is an identity mapping and no store round-trip is
    needed.

    This is the single seam for renaming: when the id becomes opaque and the name
    mutable, replace the body with a real ``(account_id, name) -> namespace_id``
    lookup against the indexed ``name`` column and every call site inherits it.
    (That lookup needs async access to the system store, so this function and
    ``get_request_scope`` would become async then.)
    """
    return name


@dataclass
class RequestScope:
    """The tenant scope a request addresses: account (header) + namespace (path)."""

    account_id: str
    namespace_id: str


def get_request_scope(request: Request, account_id: AccountIdDep) -> RequestScope:
    """Resolve the full ``(account_id, namespace_id)`` scope for a route.

    ``account_id`` comes from the ``X-Account-Id`` header; the namespace ``name``
    is the ``{namespace}`` URL path segment (absent on account-wide routes →
    default) and is resolved to its internal id via :func:`resolve_namespace_id`.
    """
    name = request.path_params.get("namespace") or DEFAULT_NAMESPACE
    return RequestScope(
        account_id=account_id,
        namespace_id=resolve_namespace_id(account_id, name),
    )


def get_system_store(request: Request) -> SystemStore:
    return request.app.state.system_store  # type: ignore[no-any-return]


def get_app_cache(request: Request) -> AppCache:
    return request.app.state.app_cache  # type: ignore[no-any-return]


def get_system_resources(request: Request) -> SystemResources:
    return request.app.state.system_resources  # type: ignore[no-any-return]


def get_skill_registry(request: Request) -> SkillRegistry:
    return request.app.state.skill_registry  # type: ignore[no-any-return]


def get_skill_bundle_store(request: Request) -> SkillBundleStore:
    store = request.app.state.skill_bundle_store
    if store is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Skill uploads require a system document store; none is configured.",
        )
    return store  # type: ignore[no-any-return]


AccountIdDep = Annotated[str, Depends(get_account_id)]
RequestScopeDep = Annotated[RequestScope, Depends(get_request_scope)]
SystemStoreDep = Annotated[SystemStore, Depends(get_system_store)]
AppCacheDep = Annotated[AppCache, Depends(get_app_cache)]
SystemResourcesDep = Annotated[SystemResources, Depends(get_system_resources)]
SkillRegistryDep = Annotated[SkillRegistry, Depends(get_skill_registry)]
SkillBundleStoreDep = Annotated[SkillBundleStore, Depends(get_skill_bundle_store)]
