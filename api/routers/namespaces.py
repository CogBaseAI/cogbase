"""CRUD endpoints for managing namespaces within an account.

A namespace is an in-account organizational unit: applications, skills, and all
other resources are addressed as ``/namespaces/{namespace}/...``.  The account is
the security boundary (the ``X-Account-Id`` header); the namespace is a handle
that is only unique within an account.  These endpoints manage the namespace
metadata records themselves — a namespace holding apps is also auto-registered on
app creation so it always surfaces in the listing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from api.dependencies import (
    AccountIdDep,
    SystemStoreDep,
    resolve_namespace_id,
)
from api.models import (
    CreateNamespaceRequest,
    NamespaceListResponse,
    NamespaceResponse,
    UpdateNamespaceRequest,
)
from api.system_store import NamespaceRecord
from cogbase.stores.schema import validate_resource_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/namespaces", tags=["namespaces"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_response(record: NamespaceRecord) -> NamespaceResponse:
    return NamespaceResponse(
        account_id=record.account_id,
        name=record.name,
        description=record.description,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post("", response_model=NamespaceResponse, status_code=status.HTTP_201_CREATED)
async def create_namespace(
    account_id: AccountIdDep,
    system_store: SystemStoreDep,
    body: CreateNamespaceRequest,
) -> NamespaceResponse:
    """Create a namespace in the calling account."""
    try:
        name = validate_resource_name(body.name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    # The record carries both a stable internal ``namespace_id`` and the
    # user-facing ``name``. They coincide today (the name is minted as the id);
    # when rename lands, ``namespace_id`` becomes an opaque uuid generated here
    # and only ``resolve_namespace_id`` (the name→id seam) has to change.
    namespace_id = name

    if await system_store.get_namespace(account_id, namespace_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Namespace '{name}' already exists",
        )

    now = _now()
    record = NamespaceRecord(
        account_id=account_id,
        namespace_id=namespace_id,
        name=name,
        description=body.description,
        created_at=now,
        updated_at=now,
    )
    await system_store.save_namespace(record)
    logger.info("Created namespace '%s' (account=%s)", namespace_id, account_id)
    return _to_response(record)


@router.get("", response_model=NamespaceListResponse)
async def list_namespaces(
    account_id: AccountIdDep,
    system_store: SystemStoreDep,
) -> NamespaceListResponse:
    """List every namespace in the calling account, most-recently-created first."""
    records = await system_store.list_namespaces(account_id)
    items = [_to_response(r) for r in records]
    return NamespaceListResponse(namespaces=items, total=len(items))


@router.get("/{namespace}", response_model=NamespaceResponse)
async def get_namespace(
    namespace: str,
    account_id: AccountIdDep,
    system_store: SystemStoreDep,
) -> NamespaceResponse:
    """Return metadata for a single namespace."""
    namespace_id = resolve_namespace_id(account_id, namespace)
    record = await system_store.get_namespace(account_id, namespace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Namespace '{namespace}' not found")
    return _to_response(record)


@router.patch("/{namespace}", response_model=NamespaceResponse)
async def update_namespace(
    namespace: str,
    account_id: AccountIdDep,
    system_store: SystemStoreDep,
    body: UpdateNamespaceRequest,
) -> NamespaceResponse:
    """Update a namespace's description.

    The namespace ``name`` is its identity and is not mutable — only the
    description can change.
    """
    if body.description is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'description' must be provided",
        )

    namespace_id = resolve_namespace_id(account_id, namespace)
    record = await system_store.get_namespace(account_id, namespace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Namespace '{namespace}' not found")

    updates: dict = {"updated_at": _now()}
    if body.description is not None:
        updates["description"] = body.description
    updated = record.model_copy(update=updates)
    await system_store.save_namespace(updated)
    logger.info("Updated namespace '%s' (account=%s)", namespace, account_id)
    return _to_response(updated)


@router.delete("/{namespace}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_namespace(
    namespace: str,
    account_id: AccountIdDep,
    system_store: SystemStoreDep,
) -> None:
    """Delete an empty namespace.

    A namespace that still holds applications is refused with 409 — delete or
    move its apps first.
    """
    namespace_id = resolve_namespace_id(account_id, namespace)
    record = await system_store.get_namespace(account_id, namespace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Namespace '{namespace}' not found")

    apps = await system_store.list_apps(account_id, namespace_id)
    if apps:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Namespace '{namespace}' still contains {len(apps)} application(s); "
                "delete them before deleting the namespace"
            ),
        )

    await system_store.delete_namespace(account_id, namespace_id)
    logger.info("Deleted namespace '%s' (account=%s)", namespace_id, account_id)
