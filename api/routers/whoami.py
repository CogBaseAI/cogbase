"""Identity bootstrap endpoint the UI calls on load.

``GET /whoami`` returns the account the server resolved for the request plus the
deployment mode, so the UI never has to source a tenant itself. In ``saas`` mode
the account and user are derived from a verified Bearer access token (the
authoritative resolver); a caller with no valid token gets ``account_id: null``,
which the UI reads as "show the login screen". In ``dev`` mode the account still
comes from the ``X-Account-Id`` header (trust-on-declaration).

This endpoint is deliberately reachable *without* a token so the UI can discover
the deployment mode before authenticating — it never raises 401 itself.

Named ``/whoami`` rather than ``/session`` to avoid colliding with CogBase's
conversational *session* concept (episodic/short-term memory).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header

from api.dependencies import (
    DEFAULT_ACCOUNT_ID,
    get_deployment_mode,
    principal_claims,
)
from api.models import WhoAmIResponse

router = APIRouter(tags=["identity"])


@router.get("/whoami", response_model=WhoAmIResponse, response_model_exclude_none=True)
async def whoami(
    x_account_id: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
    mode: str = Depends(get_deployment_mode),
) -> WhoAmIResponse:
    """Return the resolved account, user, and deployment mode for the caller."""
    if mode == "saas":
        claims = principal_claims(authorization)
        if claims is None:
            return WhoAmIResponse(account_id=None, mode=mode)
        return WhoAmIResponse(
            account_id=claims.get("account_id"),
            mode=mode,
            user_id=claims.get("sub"),
            email=claims.get("email"),
            role=claims.get("role"),
        )
    return WhoAmIResponse(account_id=x_account_id or DEFAULT_ACCOUNT_ID, mode=mode)
