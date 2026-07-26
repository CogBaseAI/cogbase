"""First-party email/password authentication endpoints.

Signup / login issue an access token (short-lived HS256 JWT) plus a refresh token
(opaque, DB-backed, revocable). Every other route derives its tenant from the
verified access token (see ``api/dependencies.py``), so these endpoints are the
only ones reachable without one.

Account model: the first user to sign up without an invite mints a new account and
becomes its ``owner``; teammates join an existing account by redeeming an invite
token (``POST /auth/invite`` → the invitee signs up with that token). This backs
the invite-only pilot while keeping open signup one config flip away.

A freshly-minted account is seeded with a default starter workspace — a
``legal-team`` namespace holding a ``contract-analyst`` application (no documents
ingested) — so the owner lands on something usable. See
``api/provisioning.py``. Provisioning is best-effort and never fails the signup.
"""

from __future__ import annotations

import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from api.auth import (
    REFRESH_TTL_SECONDS,
    InvalidToken,
    create_access_token,
    decode_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from api.dependencies import AppCacheDep, SystemResourcesDep, SystemStoreDep
from api.provisioning import provision_default_workspace
from api.models import (
    AccessTokenResponse,
    InviteRequest,
    InviteResponse,
    LoginRequest,
    LogoutRequest,
    LogoutResponse,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
)
from api.system_store import (
    AccountRecord,
    InviteRecord,
    RefreshTokenRecord,
    SystemStore,
    UserRecord,
    new_account_id,
    new_user_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Pragmatic email shape check — avoids pulling in the email-validator dependency
# while rejecting obvious garbage. Real deliverability is proven by the invite flow.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_INVITE_TTL = timedelta(days=7)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp back to an aware datetime."""
    return datetime.fromisoformat(ts)


async def _issue_tokens(store: SystemStore, user: UserRecord) -> TokenResponse:
    """Mint an access + refresh token pair for a user and persist the refresh row."""
    access = create_access_token(
        user_id=user.user_id,
        account_id=user.account_id,
        email=user.email,
        role=user.role,
    )
    refresh = generate_refresh_token()
    now = _now()
    await store.save_refresh_token(RefreshTokenRecord(
        token_hash=hash_refresh_token(refresh),
        user_id=user.user_id,
        account_id=user.account_id,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=REFRESH_TTL_SECONDS)).isoformat(),
    ))
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        account_id=user.account_id,
        user_id=user.user_id,
        email=user.email,
        role=user.role,
    )


async def get_current_principal(
    system_store: SystemStoreDep,
    authorization: Annotated[str | None, Header()] = None,
) -> UserRecord:
    """Resolve the authenticated user from a Bearer access token.

    A route dependency for endpoints that need the full principal (e.g. an
    owner-only action). Raises 401 when the token is missing/invalid/expired or
    the user no longer exists.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = decode_token(token)
    except InvalidToken as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await system_store.get_user_by_id(claims.get("sub", ""))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown principal"
        )
    return user


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    body: SignupRequest,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    system_resources: SystemResourcesDep,
) -> TokenResponse:
    """Create a user. Without an invite, mint a new account and become its owner."""
    email = _normalize_email(body.email)
    if not _EMAIL_RE.match(email):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid email address",
        )
    if await system_store.get_user_by_email(email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    now = _now()
    if body.invite_token:
        invite = await system_store.get_invite(body.invite_token)
        if (
            invite is None
            or invite.accepted_at is not None
            or _parse_ts(invite.expires_at) <= now
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invite is invalid or has expired",
            )
        if _normalize_email(invite.email) != email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invite was issued for a different email",
            )
        account_id = invite.account_id
        role = invite.role
    else:
        # New org: mint an account and make this user its owner.
        account_id = new_account_id()
        role = "owner"
        await system_store.save_account(AccountRecord(
            account_id=account_id,
            name=email.split("@", 1)[0],
            created_at=now.isoformat(),
        ))

    user = UserRecord(
        user_id=new_user_id(),
        email=email,
        password_hash=hash_password(body.password),
        account_id=account_id,
        role=role,
        created_at=now.isoformat(),
    )
    await system_store.save_user(user)
    if body.invite_token:
        await system_store.mark_invite_accepted(body.invite_token)
    else:
        # Fresh account: seed a starter workspace (legal-team namespace +
        # contract-analyst app). Best-effort — never fails the signup.
        await provision_default_workspace(
            account_id,
            system_store=system_store,
            app_cache=app_cache,
            system_resources=system_resources,
        )
    logger.info("user signed up user_id=%s account=%s role=%s", user.user_id, account_id, role)
    return await _issue_tokens(system_store, user)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, system_store: SystemStoreDep) -> TokenResponse:
    """Verify credentials and issue tokens."""
    email = _normalize_email(body.email)
    user = await system_store.get_user_by_email(email)
    # Same error whether the user is unknown or the password is wrong, so the
    # endpoint doesn't leak which emails are registered.
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    return await _issue_tokens(system_store, user)


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(body: RefreshRequest, system_store: SystemStoreDep) -> AccessTokenResponse:
    """Exchange a valid refresh token for a fresh access token."""
    record = await system_store.get_refresh_token(hash_refresh_token(body.refresh_token))
    if (
        record is None
        or record.revoked_at is not None
        or _parse_ts(record.expires_at) <= _now()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid, revoked, or expired",
        )
    user = await system_store.get_user_by_id(record.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown principal"
        )
    access = create_access_token(
        user_id=user.user_id,
        account_id=user.account_id,
        email=user.email,
        role=user.role,
    )
    return AccessTokenResponse(access_token=access)


@router.post("/logout", response_model=LogoutResponse)
async def logout(body: LogoutRequest, system_store: SystemStoreDep) -> LogoutResponse:
    """Revoke a refresh token (idempotent)."""
    await system_store.revoke_refresh_token(hash_refresh_token(body.refresh_token))
    return LogoutResponse()


@router.post("/invite", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
async def invite(
    body: InviteRequest,
    system_store: SystemStoreDep,
    principal: Annotated[UserRecord, Depends(get_current_principal)],
) -> InviteResponse:
    """Issue an invite token for an email onto the caller's account (owner only)."""
    if principal.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only an account owner can issue invites",
        )
    email = _normalize_email(body.email)
    if not _EMAIL_RE.match(email):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid email address",
        )
    if await system_store.get_user_by_email(email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )
    now = _now()
    record = InviteRecord(
        token=secrets.token_urlsafe(24),
        account_id=principal.account_id,
        email=email,
        role=body.role,
        created_at=now.isoformat(),
        expires_at=(now + _INVITE_TTL).isoformat(),
    )
    await system_store.save_invite(record)
    logger.info("invite issued account=%s email=%s role=%s", principal.account_id, email, body.role)
    return InviteResponse(
        token=record.token,
        email=record.email,
        account_id=record.account_id,
        role=record.role,
        expires_at=record.expires_at,
    )
