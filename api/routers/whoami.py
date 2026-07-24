"""Identity bootstrap endpoint the UI calls on load.

``GET /whoami`` returns the account the server resolved for the request plus the
deployment mode, so the UI never has to source a tenant itself. Today the account
comes from the ``X-Account-Id`` header (trust-on-declaration); this endpoint is the
stable seam that becomes the authoritative resolver once auth binds the account to
an authenticated principal — the UI contract does not change when it does.

Named ``/whoami`` rather than ``/session`` to avoid colliding with CogBase's
conversational *session* concept (episodic/short-term memory).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import AccountIdDep, get_deployment_mode
from api.models import WhoAmIResponse

router = APIRouter(tags=["identity"])


@router.get("/whoami", response_model=WhoAmIResponse)
async def whoami(
    account_id: AccountIdDep,
    mode: str = Depends(get_deployment_mode),
) -> WhoAmIResponse:
    """Return the resolved account and deployment mode for the calling request."""
    return WhoAmIResponse(account_id=account_id, mode=mode)
