"""Tests for CogBaseApp's session lifecycle (start / close → distill).

Covers the wiring seam between the app and the memory tiers: closing a session
evicts the short-term cache and, when a distiller is wired, runs distillation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.app import CogBaseApp


def _app(*, short_term=None, distiller=None) -> CogBaseApp:
    return CogBaseApp(
        "testapp",
        [],
        MagicMock(),
        app_id="app1",
        document_store=MagicMock(),
        structured_store=MagicMock(),
        workflow_runners={},
        llm=MagicMock(),
        task_store=MagicMock(),
        short_term=short_term,
        distiller=distiller,
    )


@pytest.mark.asyncio
async def test_start_session_requires_short_term():
    app = _app(short_term=None)
    with pytest.raises(RuntimeError):
        await app.start_session()


@pytest.mark.asyncio
async def test_start_session_delegates_to_short_term():
    short_term = MagicMock()
    short_term.start_session = AsyncMock(return_value="sess-1")
    app = _app(short_term=short_term)

    sid = await app.start_session()
    assert sid == "sess-1"
    short_term.start_session.assert_awaited_once()
    # app_id is stamped from the app, not the caller.
    assert short_term.start_session.call_args.kwargs["app_id"] == "app1"


@pytest.mark.asyncio
async def test_close_session_evicts_cache_and_distills():
    short_term = MagicMock()
    short_term.end_session = AsyncMock()
    distiller = MagicMock()
    distiller.distill_session = AsyncMock(return_value=["m1"])
    app = _app(short_term=short_term, distiller=distiller)

    ran = await app.close_session("sess-1")
    assert ran is True
    short_term.end_session.assert_awaited_once_with("sess-1")
    distiller.distill_session.assert_awaited_once_with(session_id="sess-1")


@pytest.mark.asyncio
async def test_close_session_without_distiller_just_evicts():
    short_term = MagicMock()
    short_term.end_session = AsyncMock()
    app = _app(short_term=short_term, distiller=None)

    ran = await app.close_session("sess-1")
    assert ran is False
    short_term.end_session.assert_awaited_once_with("sess-1")
