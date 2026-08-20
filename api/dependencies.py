"""FastAPI dependency providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from api.app_cache import AppCache
from api.auth import InvalidToken, decode_token
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.core.onboarding import InterviewSkillResolver
from cogbase.core.profile import AccountProfileStore
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

#: How this instance resolves the calling account, set by the operator through
#: ``SystemConfig.deployment_mode`` (the system YAML) and applied at startup via
#: :func:`set_deployment_mode`. Also advisory metadata the UI reads from
#: ``GET /whoami`` to decide whether to expose an account switcher:
#:   - ``dev``: account is trust-on-declaration via the X-Account-Id header, so
#:     the UI keeps an editable account field.
#:   - ``saas`` / ``single_tenant``: the account is server-authoritative
#:     (derived from a verified access token), so the UI treats the account
#:     returned by /whoami as read-only.
#:
#: This module-level value is only the *pre-startup* fallback — the value before
#: ``lifespan`` has run, i.e. bare-ASGI unit tests that drive the app without
#: booting it. It is intentionally the permissive ``dev`` so those paths need no
#: token. Any real deployment boots through ``lifespan``, which calls
#: :func:`set_deployment_mode` with the config value (fail-secure ``saas`` when
#: the YAML omits the key).
_deployment_mode = "dev"


def set_deployment_mode(mode: str) -> None:
    """Apply the operator-declared deployment mode resolved from system config.

    Called once from ``lifespan`` at startup. Overrides the pre-startup ``dev``
    fallback with the configured mode for the lifetime of the process.
    """
    global _deployment_mode
    _deployment_mode = mode


def get_deployment_mode() -> str:
    """Return the active deployment mode (see :data:`_deployment_mode`)."""
    return _deployment_mode


#: Whether an authenticated account may upload/replace/delete its own skills,
#: set by the operator through ``SystemConfig.tenant_skill_upload`` and applied
#: at startup via :func:`set_tenant_skill_upload`. Defaults closed (``False``)
#: — unlike :data:`_deployment_mode`, this has no permissive pre-startup
#: fallback: the safe default is what a bare-ASGI unit test gets too, so a test
#: exercising upload/replace/delete must opt in explicitly via
#: :func:`set_tenant_skill_upload` rather than relying on an ergonomic default.
_tenant_skill_upload = False


def set_tenant_skill_upload(enabled: bool) -> None:
    """Apply the operator-declared tenant-skill-upload flag from system config."""
    global _tenant_skill_upload
    _tenant_skill_upload = enabled


def get_tenant_skill_upload() -> bool:
    """Whether tenant skill upload is enabled (see :data:`_tenant_skill_upload`)."""
    return _tenant_skill_upload


#: Whether ``PATCH /system/config`` may live-patch the system LLM/embedding
#: config, set by the operator through ``SystemConfig.system_config_writable``
#: and applied at startup via :func:`set_system_config_writable`. Defaults
#: closed (``False``) for the same reason as :data:`_tenant_skill_upload`: the
#: route is unauthenticated by construction, so the safe default has no
#: permissive pre-startup fallback.
_system_config_writable = False


def set_system_config_writable(enabled: bool) -> None:
    """Apply the operator-declared system-config-writable flag from system config."""
    global _system_config_writable
    _system_config_writable = enabled


def get_system_config_writable() -> bool:
    """Whether ``PATCH /system/config`` is enabled (see :data:`_system_config_writable`)."""
    return _system_config_writable


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
    - any other mode (``dev`` and, for now, ``single_tenant``): the
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


def get_interview_skill_resolver(request: Request) -> InterviewSkillResolver | None:
    """The deployment's per-account onboarding-interview resolver, if it set one.

    ``getattr`` rather than attribute access because this one is optional and
    genuinely absent in most deployments: a single-vertical process wants the
    process-wide ``COGBASE_INTERVIEW_SKILL`` and installs nothing here. An
    embedder that serves several verticals sets ``app.state.interview_skill_resolver``
    during startup — see :data:`cogbase.core.onboarding.InterviewSkillResolver`.
    """
    return getattr(request.app.state, "interview_skill_resolver", None)


def get_skill_bundle_store(request: Request) -> SkillBundleStore:
    store = request.app.state.skill_bundle_store
    if store is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Skill uploads require a system document store; none is configured.",
        )
    return store  # type: ignore[no-any-return]


def get_account_profile_store(
    resources: Annotated[SystemResources, Depends(get_system_resources)],
) -> AccountProfileStore:
    """Provide the account company-profile store over the *system* document store.

    The profile is account-scoped — shared by every namespace and app — so it is
    read and written through the unscoped system store, which
    :class:`AccountProfileStore` narrows per call. A deployment without a system
    document store cannot hold profiles at all, so the route is 503 rather than
    silently profile-less (the read path in ``build_app`` degrades quietly; this
    write path must not).
    """
    store = resources.document_store
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Company profiles require a system document store; none is configured.",
        )
    return AccountProfileStore(store)


AccountIdDep = Annotated[str, Depends(get_account_id)]
RequestScopeDep = Annotated[RequestScope, Depends(get_request_scope)]
SystemStoreDep = Annotated[SystemStore, Depends(get_system_store)]
AppCacheDep = Annotated[AppCache, Depends(get_app_cache)]
SystemResourcesDep = Annotated[SystemResources, Depends(get_system_resources)]
SkillRegistryDep = Annotated[SkillRegistry, Depends(get_skill_registry)]
SkillBundleStoreDep = Annotated[SkillBundleStore, Depends(get_skill_bundle_store)]
InterviewSkillResolverDep = Annotated[
    InterviewSkillResolver | None, Depends(get_interview_skill_resolver)
]
AccountProfileStoreDep = Annotated[AccountProfileStore, Depends(get_account_profile_store)]
