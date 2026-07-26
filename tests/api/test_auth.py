"""Integration tests for first-party email/password auth and saas-mode enforcement.

Uses httpx.AsyncClient against the FastAPI app with an in-memory SystemStore.
Covers the auth endpoints (signup/login/refresh/logout/invite) and the critical
security property: in ``saas`` mode the tenant comes from a verified token, and a
token for one account cannot read another account's data.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.app_cache import AppCache
from api.dependencies import (
    get_app_cache,
    get_deployment_mode,
    get_skill_registry,
    get_system_resources,
    get_system_store,
)
from api.main import app
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.stores.structured.memory import InMemoryStructuredStore


@pytest_asyncio.fixture
async def ctx():
    """AsyncClient plus the underlying SystemStore, with a mode setter."""
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: AppCache()
    app.dependency_overrides[get_system_resources] = lambda: SystemResources(
        structured_store=InMemoryStructuredStore()
    )
    app.dependency_overrides[get_skill_registry] = lambda: SkillRegistry()

    def set_mode(mode: str) -> None:
        app.dependency_overrides[get_deployment_mode] = lambda: mode

    set_mode("dev")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {"client": client, "system_store": system_store, "set_mode": set_mode}

    app.dependency_overrides.clear()


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Signup / login / refresh / logout
# ---------------------------------------------------------------------------


class TestSignupLogin:
    @pytest.mark.asyncio
    async def test_signup_mints_account_and_owner(self, ctx):
        resp = await ctx["client"].post(
            "/auth/signup", json={"email": "Owner@Acme.co", "password": "hunter2hunter"}
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["role"] == "owner"
        assert body["email"] == "owner@acme.co"  # normalized
        assert body["account_id"].startswith("acct_")
        assert body["access_token"] and body["refresh_token"]

    @pytest.mark.asyncio
    async def test_duplicate_email_rejected(self, ctx):
        payload = {"email": "dup@acme.co", "password": "hunter2hunter"}
        assert (await ctx["client"].post("/auth/signup", json=payload)).status_code == 201
        resp = await ctx["client"].post("/auth/signup", json=payload)
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_signup_provisions_default_workspace(self, ctx):
        """A fresh account lands with a legal-team namespace + contract-analyst app."""
        auth = (await ctx["client"].post(
            "/auth/signup", json={"email": "founder@acme.co", "password": "hunter2hunter"}
        )).json()
        # dev mode resolves the account from X-Account-Id, so address the freshly
        # minted account explicitly.
        headers = {"X-Account-Id": auth["account_id"]}

        namespaces = (await ctx["client"].get("/namespaces", headers=headers)).json()
        assert {n["name"] for n in namespaces["namespaces"]} == {"legal-team"}

        apps = (await ctx["client"].get(
            "/namespaces/legal-team/applications", headers=headers
        )).json()
        assert [a["name"] for a in apps["applications"]] == ["contract-analyst"]

    @pytest.mark.asyncio
    async def test_invited_member_does_not_reprovision(self, ctx):
        """Joining via invite adds no second starter workspace to the account."""
        owner = (await ctx["client"].post(
            "/auth/signup", json={"email": "owner@team.co", "password": "hunter2hunter"}
        )).json()
        invite = (await ctx["client"].post(
            "/auth/invite",
            json={"email": "member@team.co", "role": "member"},
            headers=_bearer(owner["access_token"]),
        )).json()
        member = (await ctx["client"].post(
            "/auth/signup",
            json={"email": "member@team.co", "password": "hunter2hunter", "invite_token": invite["token"]},
        )).json()
        assert member["account_id"] == owner["account_id"]

        namespaces = (await ctx["client"].get(
            "/namespaces", headers={"X-Account-Id": owner["account_id"]}
        )).json()
        # Still exactly one legal-team namespace — the invite path skips provisioning.
        assert [n["name"] for n in namespaces["namespaces"]] == ["legal-team"]

    @pytest.mark.asyncio
    async def test_short_password_rejected(self, ctx):
        resp = await ctx["client"].post(
            "/auth/signup", json={"email": "x@acme.co", "password": "short"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_login_happy_and_wrong_password(self, ctx):
        await ctx["client"].post(
            "/auth/signup", json={"email": "u@acme.co", "password": "hunter2hunter"}
        )
        ok = await ctx["client"].post(
            "/auth/login", json={"email": "u@acme.co", "password": "hunter2hunter"}
        )
        assert ok.status_code == 200 and ok.json()["access_token"]

        bad = await ctx["client"].post(
            "/auth/login", json={"email": "u@acme.co", "password": "wrongpassword"}
        )
        assert bad.status_code == 401

        unknown = await ctx["client"].post(
            "/auth/login", json={"email": "nobody@acme.co", "password": "hunter2hunter"}
        )
        assert unknown.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_and_logout(self, ctx):
        signup = (await ctx["client"].post(
            "/auth/signup", json={"email": "r@acme.co", "password": "hunter2hunter"}
        )).json()
        refresh_token = signup["refresh_token"]

        good = await ctx["client"].post("/auth/refresh", json={"refresh_token": refresh_token})
        assert good.status_code == 200 and good.json()["access_token"]

        bogus = await ctx["client"].post("/auth/refresh", json={"refresh_token": "nope"})
        assert bogus.status_code == 401

        logout = await ctx["client"].post("/auth/logout", json={"refresh_token": refresh_token})
        assert logout.status_code == 200
        after = await ctx["client"].post("/auth/refresh", json={"refresh_token": refresh_token})
        assert after.status_code == 401  # revoked


# ---------------------------------------------------------------------------
# Invites
# ---------------------------------------------------------------------------


class TestInvites:
    @pytest.mark.asyncio
    async def test_owner_invites_member_joins_same_account(self, ctx):
        owner = (await ctx["client"].post(
            "/auth/signup", json={"email": "boss@acme.co", "password": "hunter2hunter"}
        )).json()

        inv = await ctx["client"].post(
            "/auth/invite",
            json={"email": "teammate@acme.co"},
            headers=_bearer(owner["access_token"]),
        )
        assert inv.status_code == 201, inv.text
        token = inv.json()["token"]

        joined = await ctx["client"].post(
            "/auth/signup",
            json={"email": "teammate@acme.co", "password": "hunter2hunter", "invite_token": token},
        )
        assert joined.status_code == 201
        body = joined.json()
        assert body["role"] == "member"
        assert body["account_id"] == owner["account_id"]  # same tenant

    @pytest.mark.asyncio
    async def test_invite_email_mismatch_rejected(self, ctx):
        owner = (await ctx["client"].post(
            "/auth/signup", json={"email": "boss2@acme.co", "password": "hunter2hunter"}
        )).json()
        token = (await ctx["client"].post(
            "/auth/invite",
            json={"email": "invited@acme.co"},
            headers=_bearer(owner["access_token"]),
        )).json()["token"]

        resp = await ctx["client"].post(
            "/auth/signup",
            json={"email": "someoneelse@acme.co", "password": "hunter2hunter", "invite_token": token},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invite_single_use(self, ctx):
        owner = (await ctx["client"].post(
            "/auth/signup", json={"email": "boss3@acme.co", "password": "hunter2hunter"}
        )).json()
        token = (await ctx["client"].post(
            "/auth/invite",
            json={"email": "once@acme.co"},
            headers=_bearer(owner["access_token"]),
        )).json()["token"]

        first = await ctx["client"].post(
            "/auth/signup",
            json={"email": "once@acme.co", "password": "hunter2hunter", "invite_token": token},
        )
        assert first.status_code == 201
        # Re-using the (now accepted) invite fails — email is already taken and
        # the invite is spent.
        second = await ctx["client"].post(
            "/auth/signup",
            json={"email": "once@acme.co", "password": "hunter2hunter", "invite_token": token},
        )
        assert second.status_code == 409

    @pytest.mark.asyncio
    async def test_member_cannot_invite(self, ctx):
        owner = (await ctx["client"].post(
            "/auth/signup", json={"email": "owner4@acme.co", "password": "hunter2hunter"}
        )).json()
        token = (await ctx["client"].post(
            "/auth/invite",
            json={"email": "member4@acme.co"},
            headers=_bearer(owner["access_token"]),
        )).json()["token"]
        member = (await ctx["client"].post(
            "/auth/signup",
            json={"email": "member4@acme.co", "password": "hunter2hunter", "invite_token": token},
        )).json()

        resp = await ctx["client"].post(
            "/auth/invite",
            json={"email": "another@acme.co"},
            headers=_bearer(member["access_token"]),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_invite_requires_auth(self, ctx):
        resp = await ctx["client"].post("/auth/invite", json={"email": "x@acme.co"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# saas-mode enforcement + cross-tenant isolation (the critical property)
# ---------------------------------------------------------------------------


class TestSaasEnforcement:
    @pytest.mark.asyncio
    async def test_protected_route_requires_token_in_saas(self, ctx):
        ctx["set_mode"]("saas")
        # No token, and even a forged account header, is rejected.
        resp = await ctx["client"].get("/namespaces")
        assert resp.status_code == 401
        forged = await ctx["client"].get("/namespaces", headers={"X-Account-Id": "victim"})
        assert forged.status_code == 401

    @pytest.mark.asyncio
    async def test_whoami_reports_saas_without_token(self, ctx):
        ctx["set_mode"]("saas")
        resp = await ctx["client"].get("/whoami")
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "saas"
        assert "account_id" not in body  # excluded when null → UI shows login

    @pytest.mark.asyncio
    async def test_token_scopes_to_its_own_account(self, ctx):
        client = ctx["client"]
        a = (await client.post(
            "/auth/signup", json={"email": "a@a.co", "password": "hunter2hunter"}
        )).json()
        b = (await client.post(
            "/auth/signup", json={"email": "b@b.co", "password": "hunter2hunter"}
        )).json()

        ctx["set_mode"]("saas")
        # Each account creates a namespace under its own token.
        assert (await client.post(
            "/namespaces", json={"name": "alpha"}, headers=_bearer(a["access_token"])
        )).status_code == 201
        assert (await client.post(
            "/namespaces", json={"name": "beta"}, headers=_bearer(b["access_token"])
        )).status_code == 201

        # A sees only its own namespaces (its own + the default starter
        # workspace); B's are invisible — and a forged header cannot override the
        # token-derived account.
        a_list = await client.get(
            "/namespaces",
            headers={**_bearer(a["access_token"]), "X-Account-Id": b["account_id"]},
        )
        assert a_list.status_code == 200
        names = {n["name"] for n in a_list.json()["namespaces"]}
        assert names == {"alpha", "legal-team"}

        b_list = await client.get("/namespaces", headers=_bearer(b["access_token"]))
        assert {n["name"] for n in b_list.json()["namespaces"]} == {"beta", "legal-team"}

    @pytest.mark.asyncio
    async def test_dev_mode_stays_header_only(self, ctx):
        """dev mode keeps working with the X-Account-Id header, no token needed."""
        resp = await ctx["client"].get("/namespaces", headers={"X-Account-Id": "team"})
        assert resp.status_code == 200
