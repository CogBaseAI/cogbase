"""Unit tests for cogbase.memory.distill.Distiller.

The distiller is exercised over a real episodic log (local-fs backed) seeded the
way the query runner records turns, with a fake extraction LLM returning chosen
candidates.  Reconciliation runs against a real :class:`LongTermMemory` over
in-memory stores, so candidates → records end-to-end.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.memory.distill import Distiller
from cogbase.memory.episodic import EpisodicMemory
from cogbase.memory.long_term import LongTermMemory
from cogbase.memory.models import MemoryStatus
from cogbase.stores.log.local_fs import LocalFSLogStore
from cogbase.stores.scope import AppScope
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSMemoryVectorStore

from tests.memory.test_long_term import HashingEmbedding


@pytest.fixture
def episodic(tmp_path) -> EpisodicMemory:
    return EpisodicMemory(LocalFSLogStore(tmp_path))


async def _long_term(app_id="app1") -> LongTermMemory:
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


async def _seed_turn(ep, sid, q, a):
    await ep.record_user_message(session_id=sid, content=q)
    await ep.record_final_answer(session_id=sid, answer=a)
    await ep.flush(sid)


@pytest.mark.asyncio
async def test_distill_promotes_candidates_with_provenance(episodic):
    sid = "sess-1"
    await _seed_turn(episodic, sid, "I always want concise answers", "Noted.")
    await _seed_turn(episodic, sid, "I work at Acme Corp", "Got it.")

    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "user prefers concise answers", "kind": "preference",
         "source_seqs": [0]},
        {"content": "user works at Acme Corp", "kind": "fact",
         "source_seqs": [2]},
    ])
    distiller = Distiller(episodic, lt, llm)

    ids = await distiller.distill_session(session_id=sid)
    assert len(ids) == 2

    recs = {r.content: r for r in await lt._load_records(ids)}
    pref = recs["user prefers concise answers"]
    assert pref.status is MemoryStatus.ACTIVE
    # source_event_ids resolved to the real log triplet for seq 0.
    assert [r.seq for r in pref.source_event_ids] == [0]
    # snapshot copied the deciding turn's text.
    assert pref.evidence_snapshot["turns"][0]["text"] == "I always want concise answers"

    fact = recs["user works at Acme Corp"]
    assert fact.status is MemoryStatus.PENDING_REVIEW


@pytest.mark.asyncio
async def test_distill_carries_normalized_entities(episodic):
    sid = "sess-entities"
    await _seed_turn(episodic, sid, "I work at Acme Corp", "Got it.")
    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "user works at Acme Corp", "kind": "preference",
         "entities": ["Acme Corp", "acme corp"], "source_seqs": [0]},
    ])
    distiller = Distiller(episodic, lt, llm)

    ids = await distiller.distill_session(session_id=sid)
    rec = (await lt._load_records(ids))[0]
    assert rec.entities == ["acme corp"]


@pytest.mark.asyncio
async def test_distill_empty_thread_returns_nothing(episodic):
    lt = await _long_term()
    distiller = Distiller(episodic, lt, _extracting_llm([]))
    assert await distiller.distill_session(session_id="empty") == []


@pytest.mark.asyncio
async def test_distill_drops_candidate_with_bad_kind(episodic):
    sid = "sess-2"
    await _seed_turn(episodic, sid, "hello", "hi")
    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "valid one", "kind": "preference", "source_seqs": [0]},
        {"content": "broken", "kind": "nonsense", "source_seqs": [0]},
    ])
    # The bad-kind item is schema-invalid, so extraction fails validation and
    # nothing is returned — verifies we never promote a malformed candidate.
    distiller = Distiller(episodic, lt, llm)
    ids = await distiller.distill_session(session_id=sid)
    assert ids == []
