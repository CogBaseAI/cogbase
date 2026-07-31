"""The account's company profile — read, edit, and delete.

The company profile is stable org-wide context a customer supplies once (who they
are, jurisdictions, regulators, risk appetite, house style). It is *account*
scoped, not namespace scoped, so these routes carry no ``{namespace}`` segment —
like ``GET /applications``, they address the whole account.

Two stores back one resource, following the ``skill_records`` precedent: the
markdown body lives in the system document store (``AccountProfileStore``), the
edit metadata in the ``profile_records`` table. A write touches both, then pushes
the new text into the account's already-built app instances — see
:func:`apply_profile_to_live_apps`.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, status

from api.dependencies import (
    AccountIdDep,
    AccountProfileStoreDep,
    AppCacheDep,
    SystemStoreDep,
    principal_claims,
)
from api.app_cache import AppCache
from api.models import CompanyProfileResponse, UpdateCompanyProfileRequest
from api.system_store import ProfileRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["profile"])


def apply_profile_to_live_apps(
    app_cache: AppCache, account_id: str, markdown: str | None
) -> int:
    """Push an edited profile into the account's cached app instances.

    The profile is read once per app build (``api/factory.py``) and held on the
    query runner for the life of the instance, so an edit has to reach the live
    instances somehow. Hot-patching beats evicting them: the profile is a page of
    framing prose, while a cached app owns warm memory tiers, a skill registry,
    and wired workflows that a rebuild would pay for again on the next query.
    ``markdown=None`` clears the block (the delete path).

    Node-local, like the cache itself. Other nodes keep serving the previous
    profile until their app-cache TTL expires — 60s (``api/app_cache.py``). For a
    rarely-edited page of prose that window is accepted rather than engineered
    away with cross-node invalidation.

    Returns the number of instances patched (for logging).
    """
    apps = app_cache.apps_for_account(account_id)
    for app in apps:
        app.set_account_profile(markdown)
    return len(apps)


def _updated_by(authorization: str | None) -> str | None:
    """Identify the editing principal, when the request carries a valid token.

    Returns ``None`` in ``dev`` mode (header-only tenancy, no principal to name),
    which is why ``profile_records.updated_by`` is nullable.
    """
    claims = principal_claims(authorization)
    if claims is None:
        return None
    return claims.get("email") or claims.get("sub")


def _response(
    markdown: str | None, record: ProfileRecord | None
) -> CompanyProfileResponse:
    return CompanyProfileResponse(
        markdown=markdown,
        exists=markdown is not None,
        updated_at=record.updated_at if record else None,
        updated_by=record.updated_by if record else None,
        source=record.source if record else None,
    )


@router.get("", response_model=CompanyProfileResponse)
async def get_profile(
    account_id: AccountIdDep,
    profile_store: AccountProfileStoreDep,
    system_store: SystemStoreDep,
) -> CompanyProfileResponse:
    """Return the calling account's company profile.

    An account that has not been through onboarding gets ``200`` with
    ``exists: false`` and a null body — absence is a state the UI branches on,
    not an error.
    """
    markdown = await profile_store.load(account_id)
    record = await system_store.get_profile_record(account_id) if markdown else None
    return _response(markdown, record)


@router.put("", response_model=CompanyProfileResponse)
async def update_profile(
    account_id: AccountIdDep,
    profile_store: AccountProfileStoreDep,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    body: UpdateCompanyProfileRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> CompanyProfileResponse:
    """Replace the account's company profile with *markdown*.

    Oversized bodies are refused with 413: past ``MAX_PROFILE_BYTES`` the profile
    stops being framing context and starts crowding out the documents an answer
    is grounded in. The cap is enforced in the store, so the generator's
    interview writer is held to the same limit.
    """
    try:
        await profile_store.save(account_id, body.markdown)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        )

    record = await system_store.save_profile_record(
        account_id, source="manual", updated_by=_updated_by(authorization)
    )
    patched = apply_profile_to_live_apps(app_cache, account_id, body.markdown)
    logger.info(
        "updated company profile account=%s bytes=%d live_apps_patched=%d",
        account_id, len(body.markdown.encode("utf-8")), patched,
    )
    return _response(body.markdown, record)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_profile(
    account_id: AccountIdDep,
    profile_store: AccountProfileStoreDep,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
) -> None:
    """Remove the account's profile — body, index row, and the live prompt block.

    Idempotent: deleting an account that has no profile succeeds.
    """
    await profile_store.delete(account_id)
    await system_store.delete_profile_record(account_id)
    patched = apply_profile_to_live_apps(app_cache, account_id, None)
    logger.info(
        "deleted company profile account=%s live_apps_patched=%d", account_id, patched
    )
