"""Tests for CogBaseApp.add_memory — the 'ingest a conversation into memory' path.

Builds a real EpisodicMemory + LongTermMemory + Distiller (only the extraction
LLM is faked) behind a minimal CogBaseApp, so a batch of messages flows
append → distill → activate end-to-end.  The key behaviour under test is the
gate bypass: a sub-threshold fact that distillation would normally park in
``pending_review`` is activated by ``add_memory`` and so becomes recallable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.app import CogBaseApp
from cogbase.memory.distill import Distiller
from cogbase.memory.episodic import EpisodicMemory
from cogbase.memory.long_term import LongTermMemory
from cogbase.memory.models import EventType, MemoryStatus
from cogbase.stores.log.local_fs import LocalFSLogStore
from cogbase.stores.scope import AppScope
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSMemoryVectorStore

from tests.memory.test_long_term import HashingEmbedding


async def _long_term(app_id: str = "app1") -> LongTermMemory:
    svc = LongTermMemory(
        InMemoryStructuredStore().with_scope(AppScope(app_id=app_id)),
        FAISSMemoryVectorStore().with_scope(AppScope(app_id=app_id)),
        MagicMock(),
        HashingEmbedding(),
        app_id=app_id,
    )
    await svc.setup()
    return svc


def _extracting_llm(memories: list[dict]) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": json.dumps({"memories": memories})})
    return llm


def _app(episodic, long_term, distiller, app_id="app1") -> CogBaseApp:
    return CogBaseApp(
        name=app_id,
        pipelines=[],
        runner=MagicMock(),
        app_id=app_id,
        document_store=MagicMock(),
        structured_store=MagicMock(),
        workflow_runners={},
        llm=MagicMock(),
        task_store=MagicMock(),
        episodic=episodic,
        long_term=long_term,
        distiller=distiller,
    )


@pytest.mark.asyncio
async def test_add_memory_activates_subthreshold_fact_and_returns_records(tmp_path):
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    lt = await _long_term()
    # Confidence 0.7 < FACT auto-promote threshold (0.85) -> would be pending_review.
    llm = _extracting_llm([
        {"content": "Caroline works at Acme Corp", "kind": "fact",
         "source_seqs": [1], "confidence": 0.7},
    ])
    app = _app(episodic, lt, Distiller(episodic, lt, llm))

    sid, records = await app.add_memory(
        messages=[
            {"role": "user", "content": "Caroline: I just started at Acme Corp"},
            {"role": "assistant", "content": "Melanie: congrats!"},
        ],
    )

    # A session id was generated and returned, and the record came back.
    assert sid
    assert [r.content for r in records] == ["Caroline works at Acme Corp"]
    # Gate bypassed: the sub-threshold fact is ACTIVE, not pending_review …
    assert records[0].status is MemoryStatus.ACTIVE
    # … and therefore recallable.
    hits = await lt.recall(query="Where does Caroline work?")
    assert any("Acme Corp" in h.content for h in hits)


@pytest.mark.asyncio
async def test_add_memory_dates_record_by_observation_date(tmp_path):
    # The promoted long-term record is dated by the conversation's observation
    # date — pinned onto the replayed turns and carried through distillation to
    # LongTermRecord.observed_at — not by wall-clock distill time.  This is the
    # end-to-end path a replayed past dialogue takes, so a back-dated conversation
    # yields a correctly back-dated memory.
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    lt = await _long_term()
    # The fact cites seq 1 — the user turn (seq 0 is session_started) — whose
    # timestamp add_memory pins to ``obs``.
    llm = _extracting_llm([
        {"content": "Caroline works at Acme Corp", "kind": "fact",
         "source_seqs": [1], "confidence": 0.7},
    ])
    app = _app(episodic, lt, Distiller(episodic, lt, llm))

    obs = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    _, records = await app.add_memory(
        messages=[
            {"role": "user", "content": "Caroline: I just started at Acme Corp"},
            {"role": "assistant", "content": "Melanie: congrats!"},
        ],
        observation_date=obs,
    )
    assert [r.observed_at for r in records] == [obs]


@pytest.mark.asyncio
async def test_add_memory_dates_record_even_without_observation_date(tmp_path):
    # A live conversation passes no observation_date: the turns carry wall-clock
    # timestamps, so the record is still dated (observed_at is never None — the
    # invariant the required field enforces).
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "Caroline works at Acme Corp", "kind": "fact",
         "source_seqs": [1], "confidence": 0.7},
    ])
    app = _app(episodic, lt, Distiller(episodic, lt, llm))

    _, records = await app.add_memory(
        messages=[
            {"role": "user", "content": "Caroline: I just started at Acme Corp"},
            {"role": "assistant", "content": "Melanie: congrats!"},
        ],
    )
    assert records and all(r.observed_at is not None for r in records)


@pytest.mark.asyncio
async def test_add_memory_pins_observation_date_on_session_started(tmp_path):
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    lt = await _long_term()
    app = _app(episodic, lt, Distiller(episodic, lt, _extracting_llm([])))

    obs = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    sid, _ = await app.add_memory(
        messages=[{"role": "user", "content": "Caroline: hi"}],
        session_id="conv-1-s1",
        observation_date=obs,
    )
    assert sid == "conv-1-s1"

    events = await episodic.replay(session_id="conv-1-s1")
    started = next(e for e in events if e.event_type is EventType.SESSION_STARTED)
    assert started.created_at == obs


@pytest.mark.asyncio
async def test_add_memory_requires_long_term(tmp_path):
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    # No long_term / distiller wired.
    app = _app(episodic, None, None)
    with pytest.raises(RuntimeError, match="long-term memory is not configured"):
        await app.add_memory(messages=[{"role": "user", "content": "hi"}])
