"""Integration tests for the /whoami identity bootstrap endpoint.

Uses httpx.AsyncClient pointed at the FastAPI app. /whoami only depends on the
account header and the deployment mode, so no store/resource overrides are needed.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.dependencies import get_deployment_mode
from api.main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestWhoAmI:
    @pytest.mark.asyncio
    async def test_defaults_when_header_absent(self, client):
        """No account header → the default account, dev mode."""
        resp = await client.get("/whoami")
        assert resp.status_code == 200
        assert resp.json() == {"account_id": "default", "mode": "dev"}

    @pytest.mark.asyncio
    async def test_echoes_account_header(self, client):
        """The resolved account reflects the X-Account-Id header."""
        resp = await client.get("/whoami", headers={"X-Account-Id": "acct-42"})
        assert resp.status_code == 200
        assert resp.json()["account_id"] == "acct-42"

    @pytest.mark.asyncio
    async def test_reports_deployment_mode(self, client):
        """A managed deployment surfaces its mode so the UI can lock the account."""
        app.dependency_overrides[get_deployment_mode] = lambda: "demo"
        try:
            resp = await client.get("/whoami")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json()["mode"] == "demo"
