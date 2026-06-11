"""Unit tests for cogbase.memory.long_term.LongTermMemory.

These back the service with real in-memory structured + FAISS vector stores and a
deterministic feature-hashing embedder, and drive :meth:`reconcile` with a fake
LLM that returns a chosen ``ReconcileOp`` — so each operation (ADD / UPDATE /
DELETE / NOOP) and the recall rules are exercised in isolation, ahead of any
wiring into the live query path.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.embeddings.base import EmbeddingBase
from cogbase.memory.long_term import LongTermMemory
from cogbase.memory.models import (
    EventRef,
    MemoryCandidate,
    MemoryKind,
    MemoryStatus,
)
from cogbase.stores.scope import AppScope
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSMemoryVectorStore

_DIM = 32


class HashingEmbedding(EmbeddingBase):
    """Feature-hashing embedder: shared tokens → similar vectors.

    Deterministic and content-sensitive (unlike a constant stub), so FAISS
    cosine search actually ranks related memories near a query.  A constant
    final dimension keeps every vector non-zero so normalisation is well-defined.
    """

    def __init__(self, dim: int = _DIM) -> None:
        self._dim = dim

    @property
    def dimensions(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % (self._dim - 1)] += 1.0
            vec[-1] = 1.0
            out.append(vec)
        return out


def _llm_returning(payload: dict) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": json.dumps(payload)})
    return llm


async def _make_service(llm=None, *, app_id="app1") -> LongTermMemory:
    structured = InMemoryStructuredStore().with_scope(AppScope(app_id=app_id))
    vector = FAISSMemoryVectorStore().with_scope(AppScope(app_id=app_id))
    svc = LongTermMemory(
        structured,
        vector,
        llm or MagicMock(),
        HashingEmbedding(),
        app_id=app_id,
    )
    await svc.setup()
    return svc


def _candidate(content, *, kind=MemoryKind.FACT, seqs=()):
    return MemoryCandidate(
        content=content,
        kind=kind,
        source_event_ids=[EventRef(session_id="s1", seq=s, ulid=f"u{s}") for s in seqs],
        evidence_snapshot={"turns": list(seqs)},
    )


# ---------------------------------------------------------------------------
# setup / promote
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_preference_is_auto_active():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    recs = await svc._load_records([mid])
    assert recs[0].status is MemoryStatus.ACTIVE
    assert recs[0].app_id == "app1"


@pytest.mark.asyncio
async def test_promote_fact_is_gated_pending_review():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("user works at Acme", kind=MemoryKind.FACT),
    )
    recs = await svc._load_records([mid])
    assert recs[0].status is MemoryStatus.PENDING_REVIEW


# ---------------------------------------------------------------------------
# reconcile — each ReconcileOp
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconcile_add_when_no_related():
    # No existing records → reconcile takes the ADD fast-path (no LLM).
    svc = await _make_service(llm=_llm_returning({"operation": "NOOP"}))
    mid = await svc.reconcile(
        candidate=_candidate("user prefers dark mode", kind=MemoryKind.PREFERENCE),
    )
    svc._llm.complete.assert_not_called()
    assert (await svc._load_records([mid]))[0].content == "user prefers dark mode"


@pytest.mark.asyncio
async def test_reconcile_update_reinforces_confidence_and_merges_provenance():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE, seqs=[1]),
    )
    before = (await svc._load_records([mid]))[0].confidence

    svc._llm = _llm_returning({"operation": "UPDATE", "target_memory_id": mid})
    out = await svc.reconcile(
        candidate=_candidate("user likes concise answers", kind=MemoryKind.PREFERENCE, seqs=[5]),
    )
    assert out == mid
    rec = (await svc._load_records([mid]))[0]
    assert rec.confidence > before
    # provenance from both observations merged
    assert {r.seq for r in rec.source_event_ids} == {1, 5}


@pytest.mark.asyncio
async def test_reconcile_update_revises_content():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    svc._llm = _llm_returning(
        {"operation": "UPDATE", "target_memory_id": mid,
         "revised_content": "user strongly prefers concise answers with citations"}
    )
    await svc.reconcile(
        candidate=_candidate("user prefers concise answers with citations", kind=MemoryKind.PREFERENCE),
    )
    rec = (await svc._load_records([mid]))[0]
    assert rec.content == "user strongly prefers concise answers with citations"


@pytest.mark.asyncio
async def test_reconcile_delete_supersedes_when_candidate_outranks():
    svc = await _make_service()
    # Seed an inferred fact (active so it is recallable for reconcile).
    old = await svc.promote(
        candidate=_candidate("user is based in Berlin", kind=MemoryKind.FACT),
        status=MemoryStatus.ACTIVE,
    )
    # A confirmed correction contradicts it.
    svc._llm = _llm_returning({"operation": "DELETE", "target_memory_id": old})
    new = await svc.reconcile(
        candidate=_candidate("user is based in Munich", kind=MemoryKind.CORRECTION),
    )
    assert new != old
    old_rec = (await svc._load_records([old]))[0]
    new_rec = (await svc._load_records([new]))[0]
    assert old_rec.status is MemoryStatus.SUPERSEDED
    assert "Munich" in new_rec.content


@pytest.mark.asyncio
async def test_reconcile_delete_rejected_when_candidate_does_not_outrank():
    svc = await _make_service()
    # Seed a confirmed correction (high confidence).
    strong = await svc.promote(
        candidate=_candidate("user is based in Munich", kind=MemoryKind.CORRECTION),
        status=MemoryStatus.ACTIVE,
    )
    # A weaker inferred fact tries to contradict it.
    svc._llm = _llm_returning({"operation": "DELETE", "target_memory_id": strong})
    out = await svc.reconcile(
        candidate=_candidate("user is based in Berlin", kind=MemoryKind.FACT),
    )
    # Existing belief stands; no new record promoted.
    assert out == strong
    rec = (await svc._load_records([strong]))[0]
    assert rec.status is MemoryStatus.ACTIVE
    all_rows = await svc._structured.query(svc._structured_collection)
    assert len(all_rows) == 1


@pytest.mark.asyncio
async def test_reconcile_noop_leaves_record_unchanged():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    before = (await svc._load_records([mid]))[0]
    svc._llm = _llm_returning({"operation": "NOOP", "target_memory_id": mid})
    out = await svc.reconcile(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    assert out == mid
    after = (await svc._load_records([mid]))[0]
    assert after.confidence == before.confidence
    assert after.content == before.content


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_returns_active_records():
    svc = await _make_service()
    await svc.promote(
        candidate=_candidate("user prefers dark mode", kind=MemoryKind.PREFERENCE),
    )
    hits = await svc.recall(query="dark mode preference")
    assert [r.content for r in hits] == ["user prefers dark mode"]


@pytest.mark.asyncio
async def test_recall_excludes_pending_review():
    svc = await _make_service()
    # A fact is gated at pending_review by default → must not surface in recall.
    await svc.promote(
        candidate=_candidate("user works at Acme", kind=MemoryKind.FACT),
    )
    hits = await svc.recall(query="where does the user work")
    assert hits == []


@pytest.mark.asyncio
async def test_recall_isolated_across_apps():
    svc_a = await _make_service(app_id="app-a")
    svc_b = await _make_service(app_id="app-b")
    await svc_a.promote(
        candidate=_candidate("tenant a secret preference", kind=MemoryKind.PREFERENCE),
    )
    # Different app partition → must not leak.
    hits = await svc_b.recall(query="tenant a secret preference")
    assert hits == []
