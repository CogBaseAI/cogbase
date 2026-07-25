"""FastAPI dependency providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from api.app_cache import AppCache
from api.auth import InvalidToken, decode_token
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.store import SkillBundleStore


#: Default account used when a request omits the tenancy header.
#: Tenancy is logical for now — ``account_id`` is trust-on-declaration until an
#: auth layer binds the header to an authenticated principal.
#:
#: There is deliberately no default *namespace*: a namespace must be created
#: explicitly (``POST /namespaces``) before it can hold applications. Namespace-
#: less routes (e.g. account-wide listings, ``/generate/chat``) resolve only the
#: account and never address a namespace.
DEFAULT_ACCOUNT_ID = "default"

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
DEPLOYMENT_MODE = os.environ.get("COGBASE_DEPLOYMENT_MODE", "saas")


def get_deployment_mode() -> str:
    """Return the operator-declared deployment mode (see :data:`DEPLOYMENT_MODE`)."""
    return DEPLOYMENT_MODE


def principal_claims(authorization: str | None) -> dict | None:
    """Return the verified access-token claims from an ``Authorization`` header.

    Non-raising: returns ``None`` when the header is absent, not a Bearer token,
    or the token fails verification/expiry. Callers that must reject an
    unauthenticated request raise 401 themselves; ``whoami`` uses the ``None`` to
    report "unauthenticated" without erroring.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        return decode_token(token)
    except InvalidToken:
        return None


def get_account_id(
    x_account_id: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    mode: str = Depends(get_deployment_mode),
) -> str:
    """Resolve the calling tenant, enforcing auth in a managed deployment.

    - ``saas`` mode: the account is server-authoritative — derived from a verified
      Bearer access token (the ``X-Account-Id`` header is ignored). An absent or
      invalid token is rejected with 401.
    - any other mode (``dev`` and, for now, ``single_tenant`` / ``demo``): the
      account is trust-on-declaration via the ``X-Account-Id`` header, falling back
      to ``DEFAULT_ACCOUNT_ID``. This keeps local development header-only.

    The mode comes through :func:`get_deployment_mode` (not the module constant
    directly) so tests can override it via FastAPI dependency overrides.

    The account is the security boundary; the namespace is addressed in the path.
    """
    if mode == "saas":
        claims = principal_claims(authorization)
        if claims is None or not claims.get("account_id"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return claims["account_id"]
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
    is the ``{namespace}`` URL path segment, resolved to its internal id via
    :func:`resolve_namespace_id`. This dependency is for namespace-scoped routes
    only; account-wide routes take :data:`AccountIdDep` directly, since there is
    no default namespace to fall back to.
    """
    name = request.path_params.get("namespace")
    if name is None:
        # Programming error: this dependency was used on a route without a
        # ``{namespace}`` path segment. There is no default namespace to assume.
        raise RuntimeError(
            "get_request_scope requires a '{namespace}' path segment; "
            "use AccountIdDep for account-wide routes"
        )
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
