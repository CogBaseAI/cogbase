"""First-party authentication primitives: password hashing and signed tokens.

Deliberately dependency-free — everything here is built on the Python standard
library so the security boundary carries no third-party runtime dependency:

- **Passwords** are hashed with :func:`hashlib.scrypt` (memory-hard) using a
  per-password random salt; the parameters are stored inline so they can evolve
  without a migration.
- **Access tokens** are HS256 JWTs (the compact, widely understood format) signed
  with a server secret via stdlib ``hmac``; verification is constant-time.
- **Refresh tokens** are opaque random strings; only their SHA-256 hash is
  persisted (see ``api/system_store.py``), so a database read never yields a
  usable token.

The account is derived from the *verified* access-token claims, replacing the
trust-on-declaration ``X-Account-Id`` header once ``COGBASE_DEPLOYMENT_MODE`` is
a managed mode (see ``api/dependencies.py``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time

logger = logging.getLogger(__name__)

# Access tokens are short-lived; a refresh token (revocable, DB-backed) mints new
# ones so a leaked access token has a small blast radius.
ACCESS_TTL_SECONDS = 30 * 60            # 30 minutes
REFRESH_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

# scrypt cost parameters. n must be a power of two; memory use ≈ 128*n*r bytes
# (~16 MiB here). Stored inline with each hash so they can be raised later
# without invalidating existing hashes.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_MAXMEM = 128 * _SCRYPT_N * _SCRYPT_R * 2  # headroom over the ~16 MiB needed

_DEV_SECRET = "cogbase-dev-insecure-secret-change-me"
_warned_dev_secret = False


class InvalidToken(Exception):
    """Raised when an access token is malformed, tampered with, or expired."""


def get_jwt_secret() -> str:
    """Return the HMAC signing secret from ``COGBASE_JWT_SECRET``.

    Falls back to a fixed, insecure development secret (with a warning) so local
    dev and tests work out of the box. Production deployments MUST set the env
    var — the deploy runbook does.
    """
    global _warned_dev_secret
    secret = os.environ.get("COGBASE_JWT_SECRET")
    if not secret:
        if not _warned_dev_secret:
            logger.warning(
                "COGBASE_JWT_SECRET is not set — using an insecure development secret. "
                "Set COGBASE_JWT_SECRET in any real deployment."
            )
            _warned_dev_secret = True
        return _DEV_SECRET
    return secret


# ---------------------------------------------------------------------------
# base64url helpers (no padding, per the JWT spec)
# ---------------------------------------------------------------------------


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


# ---------------------------------------------------------------------------
# Password hashing (scrypt)
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a password with scrypt and a fresh random salt.

    Returns a self-describing string ``scrypt$n$r$p$salt_b64$hash_b64`` so
    :func:`verify_password` can re-derive with the exact parameters used.
    """
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN, maxmem=_SCRYPT_MAXMEM,
    )
    return "$".join([
        "scrypt", str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P),
        _b64url_encode(salt), _b64url_encode(derived),
    ])


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a stored scrypt hash."""
    try:
        scheme, n_s, r_s, p_s, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt = _b64url_decode(salt_b64)
        expected = _b64url_decode(hash_b64)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt, n=n, r=r, p=p,
            dklen=len(expected), maxmem=128 * n * r * 2,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, expected)


# ---------------------------------------------------------------------------
# Access tokens (HS256 JWT)
# ---------------------------------------------------------------------------


def _sign(signing_input: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return _b64url_encode(sig)


def create_access_token(
    *, user_id: str, account_id: str, email: str, role: str,
    ttl_seconds: int = ACCESS_TTL_SECONDS,
) -> str:
    """Mint a short-lived HS256 access token carrying the verified principal."""
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "account_id": account_id,
        "email": email,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + ttl_seconds,
    }
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    ]
    signing_input = ".".join(segments).encode("ascii")
    segments.append(_sign(signing_input, get_jwt_secret()))
    return ".".join(segments)


def decode_token(token: str) -> dict:
    """Verify an access token's signature and expiry, returning its claims.

    Raises :class:`InvalidToken` if the token is malformed, the signature does
    not match, the algorithm is not HS256, or it has expired.
    """
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        raise InvalidToken("malformed token")

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = _sign(signing_input, get_jwt_secret())
    if not hmac.compare_digest(expected_sig, sig_b64):
        raise InvalidToken("bad signature")

    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, TypeError):
        raise InvalidToken("undecodable token")

    if header.get("alg") != "HS256":
        raise InvalidToken("unexpected algorithm")

    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or time.time() >= exp:
        raise InvalidToken("expired token")

    return payload


# ---------------------------------------------------------------------------
# Refresh tokens (opaque; stored only as a hash)
# ---------------------------------------------------------------------------


def generate_refresh_token() -> str:
    """Return a fresh, high-entropy opaque refresh token."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """SHA-256 hex digest used as the storage key for a refresh token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
