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

from api.dependencies import DEFAULT_NAMESPACE, AccountIdDep, SystemStoreDep
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
        namespace_id=record.namespace_id,
        display_name=record.display_name,
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
        namespace_id = validate_resource_name(body.namespace_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    if await system_store.get_namespace(account_id, namespace_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Namespace '{namespace_id}' already exists",
        )

    now = _now()
    record = NamespaceRecord(
        account_id=account_id,
        namespace_id=namespace_id,
        display_name=body.display_name,
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
    record = await system_store.get_namespace(account_id, namespace)
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
    """Update a namespace's display name and/or description.

    The ``namespace_id`` handle is the namespace's identity and is not mutable —
    only the friendly label and description can change.
    """
    if body.display_name is None and body.description is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'display_name' or 'description' must be provided",
        )

    record = await system_store.get_namespace(account_id, namespace)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Namespace '{namespace}' not found")

    updates: dict = {"updated_at": _now()}
    if body.display_name is not None:
        updates["display_name"] = body.display_name
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

    The default namespace can't be deleted (it is the implicit fallback for
    callers that don't address a namespace), and a namespace that still holds
    applications is refused with 409 — delete or move its apps first.
    """
    if namespace == DEFAULT_NAMESPACE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The default namespace cannot be deleted",
        )

    record = await system_store.get_namespace(account_id, namespace)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Namespace '{namespace}' not found")

    apps = await system_store.list_apps(account_id, namespace)
    if apps:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Namespace '{namespace}' still contains {len(apps)} application(s); "
                "delete them before deleting the namespace"
            ),
        )

    await system_store.delete_namespace(account_id, namespace)
    logger.info("Deleted namespace '%s' (account=%s)", namespace, account_id)
