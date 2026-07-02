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
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.embeddings.base import EmbeddingBase
from cogbase.memory.long_term import MAX_REVIEW_BATCH, LongTermMemory
from cogbase.memory.models import (
    EventRef,
    MemoryCandidate,
    MemoryKind,
    MemoryStatus,
    ReconcileOp,
    ReviewDecision,
    ReviewOutcome,
)

# Sensible per-kind candidate confidence for hand-built test candidates: a
# correction outranks an inferred fact, matching the reconcile precedence rules.
_TEST_CONFIDENCE = {
    MemoryKind.CORRECTION: 0.9,
    MemoryKind.PREFERENCE: 0.7,
    MemoryKind.FACT: 0.6,
    MemoryKind.RETRIEVAL_HINT: 0.6,
}
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


# Real memory_id UUIDs are masked to small integer ids before the reconcile
# prompt and resolved back afterwards (anti-hallucination), so a fake LLM
# targets a related record by its position.  These tests each surface exactly
# one related record, so the only valid target index is 0.
_FIRST_RELATED = 0


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


# A fixed observation date for hand-built test candidates; distillation always
# supplies one (see LongTermRecord.observed_at), so fixtures must too.
_TEST_OBSERVED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _candidate(
    content, *, kind=MemoryKind.FACT, seqs=(), entities=(), confidence=None,
    observed_at=_TEST_OBSERVED_AT,
):
    return MemoryCandidate(
        content=content,
        kind=kind,
        entities=list(entities),
        source_event_ids=[EventRef(session_id="s1", seq=s, ulid=f"u{s}") for s in seqs],
        evidence_snapshot={"turns": list(seqs)},
        confidence=confidence if confidence is not None else _TEST_CONFIDENCE[kind],
        observed_at=observed_at,
    )


def test_observed_at_is_required_on_candidate_and_record():
    # observed_at is mandatory on both models: a promoted memory always derives
    # from timestamped turns, so a missing observation date is a bug we surface at
    # construction rather than silently defaulting (see LongTermRecord.observed_at).
    import pydantic

    from cogbase.memory.models import LongTermRecord

    with pytest.raises(pydantic.ValidationError):
        MemoryCandidate(content="x", kind=MemoryKind.FACT, confidence=0.6)
    with pytest.raises(pydantic.ValidationError):
        LongTermRecord(content="x", kind=MemoryKind.FACT, confidence=0.6)


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


@pytest.mark.asyncio
async def test_promote_strongly_supported_fact_is_auto_active():
    svc = await _make_service()
    # A fact at/above the fact auto-promote threshold (0.85) skips review.
    mid = await svc.promote(
        candidate=_candidate("user works at Acme", kind=MemoryKind.FACT, confidence=0.9),
    )
    assert (await svc._load_records([mid]))[0].status is MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_promote_correction_always_waits_for_review():
    svc = await _make_service()
    # Even a maximally confident correction overrides belief, so it is gated.
    mid = await svc.promote(
        candidate=_candidate("user is in Munich", kind=MemoryKind.CORRECTION, confidence=1.0),
    )
    assert (await svc._load_records([mid]))[0].status is MemoryStatus.PENDING_REVIEW


@pytest.mark.asyncio
async def test_promote_pending_skips_vector_upsert_until_accepted():
    """A gated record's vector is never read, so promote must not write it.

    Every vector reader is active-only, and ``review`` re-embeds + upserts the
    record on accept (see ``LongTermMemory._save_record``), so writing the vector
    at promote time would be pure waste that gets overwritten.  An auto-active
    preference, by contrast, is indexed immediately.
    """
    svc = await _make_service()
    real_upsert = svc._vector.upsert
    svc._vector.upsert = AsyncMock(side_effect=real_upsert)

    # Gated fact: the structured row is written, but no vector upsert.
    mid = await svc.promote(candidate=_candidate("user works at Acme"))
    assert (await svc._load_records([mid]))[0].status is MemoryStatus.PENDING_REVIEW
    svc._vector.upsert.assert_not_called()

    # Auto-active preference: indexed immediately.
    await svc.promote(
        candidate=_candidate("prefers concise answers", kind=MemoryKind.PREFERENCE)
    )
    assert svc._vector.upsert.call_count == 1

    # Review accept creates the gated record's vector, so recall now surfaces it.
    await svc.review_many(decisions=[ReviewDecision(memory_id=mid, accept=True)])
    assert svc._vector.upsert.call_count == 2
    hits = await svc.recall(query="user works at Acme")
    assert mid in [h.memory_id for h in hits]


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
async def test_embed_contents_caches_candidate_content_across_reconcile():
    # Batch-embedding a session's candidates up front and passing the cache to
    # reconcile must collapse the per-candidate double-embed (search query +
    # promote write) so each distinct content is embedded exactly once.
    svc = await _make_service(llm=_llm_returning({"operation": "NOOP"}))
    embed_calls: list[list[str]] = []
    original_embed = svc._embedder.embed

    async def _counting_embed(texts):
        embed_calls.append(list(texts))
        return await original_embed(texts)

    svc._embedder.embed = _counting_embed

    candidates = [
        _candidate("user prefers dark mode", kind=MemoryKind.PREFERENCE),
        _candidate("user works at Acme", kind=MemoryKind.FACT),
    ]
    cache = await svc.embed_contents(candidates)
    assert set(cache) == {c.content for c in candidates}
    # One batch call for both contents.
    assert embed_calls == [[c.content for c in candidates]]

    embed_calls.clear()
    for candidate in candidates:
        await svc.reconcile(candidate=candidate, embeddings=cache)
    # Two ADDs (no related records), and the candidate content for each was
    # served from the cache for both the search and the save — zero new embeds.
    assert embed_calls == []


@pytest.mark.asyncio
async def test_reconcile_update_reinforces_confidence_and_merges_provenance():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE, seqs=[1]),
    )
    before = (await svc._load_records([mid]))[0].confidence

    svc._llm = _llm_returning({"operation": "UPDATE", "target_memory_id": _FIRST_RELATED})
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
        {"operation": "UPDATE", "target_memory_id": _FIRST_RELATED,
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
    svc._llm = _llm_returning({"operation": "DELETE", "target_memory_id": _FIRST_RELATED})
    new = await svc.reconcile(
        candidate=_candidate("user is based in Munich", kind=MemoryKind.CORRECTION),
    )
    assert new != old
    old_rec = (await svc._load_records([old]))[0]
    new_rec = (await svc._load_records([new]))[0]
    assert old_rec.status is MemoryStatus.SUPERSEDED
    assert "Munich" in new_rec.content
    # The superseded record is purged from the vector index itself — not merely
    # filtered out by status — so a raw content search no longer returns it.
    raw = await svc._search_content("user is based in Berlin", top_k=50)
    assert old not in [c.doc_id for c in raw]


@pytest.mark.asyncio
async def test_reconcile_delete_rejected_when_candidate_does_not_outrank():
    svc = await _make_service()
    # Seed a confirmed correction (high confidence).
    strong = await svc.promote(
        candidate=_candidate("user is based in Munich", kind=MemoryKind.CORRECTION),
        status=MemoryStatus.ACTIVE,
    )
    # A weaker inferred fact tries to contradict it.
    svc._llm = _llm_returning({"operation": "DELETE", "target_memory_id": _FIRST_RELATED})
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
# reconcile_decided — the single-call path applies a pre-decided op, no LLM
# ---------------------------------------------------------------------------

def _decided(content, *, op, target=None, revised=None, kind=MemoryKind.FACT, **kw):
    """A candidate the extractor already decided the reconcile op for."""
    candidate = _candidate(content, kind=kind, **kw)
    candidate.operation = op
    candidate.target_memory_id = target
    candidate.revised_content = revised
    return candidate


@pytest.mark.asyncio
async def test_reconcile_decided_add_promotes_without_llm():
    # ADD is the default op; the single-call path never touches the LLM.
    svc = await _make_service(llm=_llm_returning({"operation": "NOOP"}))
    mid = await svc.reconcile_decided(
        candidate=_decided("user prefers dark mode", op=ReconcileOp.ADD,
                           kind=MemoryKind.PREFERENCE),
    )
    svc._llm.complete.assert_not_called()
    assert (await svc._load_records([mid]))[0].content == "user prefers dark mode"


@pytest.mark.asyncio
async def test_reconcile_decided_update_reinforces_target():
    svc = await _make_service(llm=_llm_returning({"operation": "NOOP"}))
    mid = await svc.promote(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE, seqs=[1]),
    )
    before = (await svc._load_records([mid]))[0].confidence

    out = await svc.reconcile_decided(
        candidate=_decided("user likes concise answers", op=ReconcileOp.UPDATE,
                           target=mid, kind=MemoryKind.PREFERENCE, seqs=[5]),
    )
    svc._llm.complete.assert_not_called()
    assert out == mid
    rec = (await svc._load_records([mid]))[0]
    assert rec.confidence > before
    assert {r.seq for r in rec.source_event_ids} == {1, 5}


@pytest.mark.asyncio
async def test_reconcile_decided_update_revises_content():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    await svc.reconcile_decided(
        candidate=_decided(
            "user prefers concise answers", op=ReconcileOp.UPDATE, target=mid,
            revised="user strongly prefers concise answers with citations",
            kind=MemoryKind.PREFERENCE,
        ),
    )
    assert (await svc._load_records([mid]))[0].content == (
        "user strongly prefers concise answers with citations"
    )


@pytest.mark.asyncio
async def test_reconcile_decided_delete_supersedes_when_candidate_outranks():
    svc = await _make_service()
    old = await svc.promote(
        candidate=_candidate("user is based in Berlin", kind=MemoryKind.FACT),
        status=MemoryStatus.ACTIVE,
    )
    new = await svc.reconcile_decided(
        candidate=_decided("user is based in Munich", op=ReconcileOp.DELETE,
                           target=old, kind=MemoryKind.CORRECTION),
    )
    assert new != old
    assert (await svc._load_records([old]))[0].status is MemoryStatus.SUPERSEDED
    assert "Munich" in (await svc._load_records([new]))[0].content


@pytest.mark.asyncio
async def test_reconcile_decided_noop_leaves_target_unchanged():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate("user prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    before = (await svc._load_records([mid]))[0]
    out = await svc.reconcile_decided(
        candidate=_decided("user prefers concise answers", op=ReconcileOp.NOOP,
                           target=mid, kind=MemoryKind.PREFERENCE),
    )
    assert out == mid
    after = (await svc._load_records([mid]))[0]
    assert after.confidence == before.confidence
    assert after.content == before.content
    # NOOP writes nothing new.
    assert len(await svc._structured.query(svc._structured_collection)) == 1


@pytest.mark.asyncio
async def test_reconcile_decided_degrades_to_add_on_missing_target():
    # An UPDATE whose target was superseded since recall must not touch the wrong
    # record — it degrades to ADD and promotes the candidate as new.
    svc = await _make_service()
    out = await svc.reconcile_decided(
        candidate=_decided("user works at Acme", op=ReconcileOp.UPDATE,
                           target="does-not-exist", kind=MemoryKind.FACT, confidence=0.9),
    )
    rec = (await svc._load_records([out]))[0]
    assert rec.content == "user works at Acme"
    assert rec.status is MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_reconcile_decided_degrades_to_add_on_inactive_target():
    # A target that exists but is no longer active (superseded) is not a valid
    # reconcile target; the op degrades to ADD.
    svc = await _make_service()
    old = await svc.promote(
        candidate=_candidate("user is based in Berlin", kind=MemoryKind.FACT),
        status=MemoryStatus.SUPERSEDED,
    )
    out = await svc.reconcile_decided(
        candidate=_decided("user is based in Munich", op=ReconcileOp.DELETE,
                           target=old, kind=MemoryKind.CORRECTION),
    )
    assert out != old
    assert "Munich" in (await svc._load_records([out]))[0].content


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
async def test_promote_normalizes_entities():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate(
            "user works at Acme Corp", kind=MemoryKind.PREFERENCE,
            entities=["Acme Corp", "  acme corp ", "User"],
        ),
    )
    rec = (await svc._load_records([mid]))[0]
    assert rec.entities == ["acme corp", "user"]


@pytest.mark.asyncio
async def test_reconcile_update_merges_entities():
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate(
            "user prefers concise answers", kind=MemoryKind.PREFERENCE,
            entities=["user"],
        ),
    )
    svc._llm = _llm_returning({"operation": "UPDATE", "target_memory_id": _FIRST_RELATED})
    await svc.reconcile(
        candidate=_candidate(
            "user likes concise answers", kind=MemoryKind.PREFERENCE,
            entities=["User", "acme corp"],
        ),
    )
    rec = (await svc._load_records([mid]))[0]
    assert rec.entities == ["user", "acme corp"]


@pytest.mark.asyncio
async def test_related_records_unions_entity_overlap_past_vector_miss():
    # A paraphrased claim about the same entity that vector search misses must
    # still surface as a reconcile candidate via the entity index.
    svc = await _make_service()
    mid = await svc.promote(
        candidate=_candidate(
            "alice's deployment target is aws", entities=["alice"],
        ),
        status=MemoryStatus.ACTIVE,
    )

    async def _no_vector_hits(query, *, top_k, embeddings=None):
        return []

    svc._search_content = _no_vector_hits
    related = await svc._related_records(
        _candidate("the rollout platform chosen is azure", entities=["Alice"])
    )
    assert [r.memory_id for r in related] == [mid]


# ---------------------------------------------------------------------------
# lookup (the pull path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lookup_by_entities_without_query():
    svc = await _make_service()
    await svc.promote(
        candidate=_candidate(
            "user works at Acme Corp", kind=MemoryKind.PREFERENCE,
            entities=["acme corp"],
        ),
    )
    await svc.promote(
        candidate=_candidate("user prefers dark mode", kind=MemoryKind.PREFERENCE),
    )
    hits = await svc.lookup(entities=["Acme Corp"])
    assert [r.content for r in hits] == ["user works at Acme Corp"]


@pytest.mark.asyncio
async def test_lookup_by_kind_without_query():
    svc = await _make_service()
    await svc.promote(
        candidate=_candidate("user prefers dark mode", kind=MemoryKind.PREFERENCE),
    )
    await svc.promote(
        candidate=_candidate("search contracts first", kind=MemoryKind.RETRIEVAL_HINT),
    )
    hits = await svc.lookup(kind=MemoryKind.RETRIEVAL_HINT)
    assert [r.content for r in hits] == ["search contracts first"]


@pytest.mark.asyncio
async def test_lookup_with_query_applies_kind_and_entity_filters():
    svc = await _make_service()
    await svc.promote(
        candidate=_candidate(
            "user prefers dark mode", kind=MemoryKind.PREFERENCE, entities=["user"],
        ),
    )
    await svc.promote(
        candidate=_candidate(
            "user prefers dark roast coffee", kind=MemoryKind.PREFERENCE,
            entities=["acme corp"],
        ),
    )
    hits = await svc.lookup(
        query="dark mode preference", kind=MemoryKind.PREFERENCE, entities=["user"],
    )
    assert [r.content for r in hits] == ["user prefers dark mode"]


@pytest.mark.asyncio
async def test_lookup_excludes_non_active():
    svc = await _make_service()
    # A fact is gated at pending_review by default → must not surface in lookup.
    await svc.promote(
        candidate=_candidate("user works at Acme Corp", entities=["acme corp"]),
    )
    assert await svc.lookup(entities=["acme corp"]) == []


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


# ---------------------------------------------------------------------------
# promotion review (pending_review -> active / superseded)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reads_tolerate_uncreated_collection():
    """Reading before the first distillation write must not raise.

    When the embedder can't report its dimensionality, ``setup`` skips eager
    creation and the collections are made lazily on the first write.  A review
    surface (the Memory tab) that lists records before any distillation has run
    must see an empty result, not a missing-collection ``KeyError``.
    """
    structured = InMemoryStructuredStore().with_scope(AppScope(app_id="app1"))
    vector = FAISSMemoryVectorStore().with_scope(AppScope(app_id="app1"))
    embedder = HashingEmbedding()
    # Force the "dimensions unknown up front" path so setup creates nothing.
    with patch.object(type(embedder), "dimensions", property(lambda self: None)):
        svc = LongTermMemory(structured, vector, MagicMock(), embedder, app_id="app1")
        await svc.setup()
        assert await svc.list_pending() == []
        assert await svc.list_records() == []
        assert await svc.lookup(kind=MemoryKind.FACT) == []


@pytest.mark.asyncio
async def test_list_pending_returns_gated_records_oldest_first():
    svc = await _make_service()
    first = await svc.promote(candidate=_candidate("user works at Acme"))
    second = await svc.promote(candidate=_candidate("user is based in Berlin"))
    # A preference auto-actives, so it must NOT appear in the pending queue.
    await svc.promote(
        candidate=_candidate("prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    pending = await svc.list_pending()
    assert [r.memory_id for r in pending] == [first, second]
    assert all(r.status is MemoryStatus.PENDING_REVIEW for r in pending)


@pytest.mark.asyncio
async def test_list_pending_filters_by_kind():
    svc = await _make_service()
    fact = await svc.promote(candidate=_candidate("user works at Acme", kind=MemoryKind.FACT))
    await svc.promote(
        candidate=_candidate("user corrected the spelling", kind=MemoryKind.CORRECTION),
    )
    pending = await svc.list_pending(kind=MemoryKind.FACT)
    assert [r.memory_id for r in pending] == [fact]


@pytest.mark.asyncio
async def test_list_records_filters_by_status():
    svc = await _make_service()
    # A fact is gated (pending_review); a preference auto-actives.
    fact = await svc.promote(candidate=_candidate("user works at Acme", kind=MemoryKind.FACT))
    pref = await svc.promote(
        candidate=_candidate("prefers concise answers", kind=MemoryKind.PREFERENCE),
    )
    active = await svc.list_records(status=MemoryStatus.ACTIVE)
    assert [r.memory_id for r in active] == [pref]
    pending = await svc.list_records(status=MemoryStatus.PENDING_REVIEW)
    assert [r.memory_id for r in pending] == [fact]
    # No status filter spans every lifecycle state.
    everything = await svc.list_records()
    assert {r.memory_id for r in everything} == {fact, pref}


@pytest.mark.asyncio
async def test_list_records_orders_by_observed_at_desc():
    svc = await _make_service()
    older = await svc.promote(candidate=_candidate(
        "older preference", kind=MemoryKind.PREFERENCE,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ))
    newer = await svc.promote(candidate=_candidate(
        "newer preference", kind=MemoryKind.PREFERENCE,
        observed_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    ))
    records = await svc.list_records(status=MemoryStatus.ACTIVE)
    assert [r.memory_id for r in records] == [newer, older]


@pytest.mark.asyncio
async def test_list_records_filters_by_kind():
    svc = await _make_service()
    fact = await svc.promote(candidate=_candidate("user works at Acme", kind=MemoryKind.FACT))
    await svc.promote(
        candidate=_candidate("user corrected the spelling", kind=MemoryKind.CORRECTION),
    )
    records = await svc.list_records(kind=MemoryKind.FACT)
    assert [r.memory_id for r in records] == [fact]


@pytest.mark.asyncio
async def test_review_accept_makes_record_recallable():
    svc = await _make_service()
    mid = await svc.promote(candidate=_candidate("user works at Acme"))
    # Gated → not recalled yet.
    assert await svc.recall(query="user works at Acme") == []

    [result] = await svc.review_many(
        decisions=[ReviewDecision(memory_id=mid, accept=True)]
    )
    assert result.outcome is ReviewOutcome.ACCEPTED
    assert (await svc._load_records([mid]))[0].status is MemoryStatus.ACTIVE
    # The vector metadata flipped too, so recall now surfaces it.
    hits = await svc.recall(query="user works at Acme")
    assert [h.memory_id for h in hits] == [mid]


@pytest.mark.asyncio
async def test_review_reject_supersedes_and_stays_hidden():
    svc = await _make_service()
    mid = await svc.promote(candidate=_candidate("user works at Acme"))
    [result] = await svc.review_many(
        decisions=[ReviewDecision(memory_id=mid, accept=False)]
    )
    assert result.outcome is ReviewOutcome.REJECTED
    assert (await svc._load_records([mid]))[0].status is MemoryStatus.SUPERSEDED
    assert await svc.recall(query="user works at Acme") == []


@pytest.mark.asyncio
async def test_review_already_decided_is_skipped_and_idempotent():
    svc = await _make_service()
    mid = await svc.promote(candidate=_candidate("user works at Acme"))
    await svc.review_many(decisions=[ReviewDecision(memory_id=mid, accept=True)])
    # A re-submitted reject must not resurrect/supersede an already-active record.
    [result] = await svc.review_many(
        decisions=[ReviewDecision(memory_id=mid, accept=False)]
    )
    assert result.outcome is ReviewOutcome.SKIPPED
    assert (await svc._load_records([mid]))[0].status is MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_review_unknown_id_is_not_found():
    svc = await _make_service()
    [result] = await svc.review_many(
        decisions=[ReviewDecision(memory_id="nope", accept=True)]
    )
    assert result.outcome is ReviewOutcome.NOT_FOUND


@pytest.mark.asyncio
async def test_review_many_applies_mixed_decisions():
    svc = await _make_service()
    accept_id = await svc.promote(candidate=_candidate("user works at Acme"))
    reject_id = await svc.promote(candidate=_candidate("user is based in Berlin"))
    results = await svc.review_many(
        decisions=[
            ReviewDecision(memory_id=accept_id, accept=True),
            ReviewDecision(memory_id=reject_id, accept=False),
            ReviewDecision(memory_id="missing", accept=True),
        ]
    )
    assert [(r.memory_id, r.outcome) for r in results] == [
        (accept_id, ReviewOutcome.ACCEPTED),
        (reject_id, ReviewOutcome.REJECTED),
        ("missing", ReviewOutcome.NOT_FOUND),
    ]
    assert (await svc._load_records([accept_id]))[0].status is MemoryStatus.ACTIVE
    assert (await svc._load_records([reject_id]))[0].status is MemoryStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_review_many_duplicate_id_decides_once():
    svc = await _make_service()
    mid = await svc.promote(candidate=_candidate("user works at Acme"))
    results = await svc.review_many(
        decisions=[
            ReviewDecision(memory_id=mid, accept=True),
            ReviewDecision(memory_id=mid, accept=False),
        ]
    )
    assert [r.outcome for r in results] == [ReviewOutcome.ACCEPTED, ReviewOutcome.SKIPPED]
    assert (await svc._load_records([mid]))[0].status is MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_review_many_over_cap_raises():
    svc = await _make_service()
    decisions = [
        ReviewDecision(memory_id=f"m{i}", accept=True)
        for i in range(MAX_REVIEW_BATCH + 1)
    ]
    with pytest.raises(ValueError):
        await svc.review_many(decisions=decisions)


# ---------------------------------------------------------------------------
# recall neighborhood traversal (the memory graph)
# ---------------------------------------------------------------------------

async def _promote_active(svc, content, *, links=()):
    return await svc.promote(
        candidate=MemoryCandidate(
            content=content,
            kind=MemoryKind.FACT,
            confidence=0.9,
            linked_memory_ids=list(links),
            observed_at=_TEST_OBSERVED_AT,
        ),
        status=MemoryStatus.ACTIVE,
    )


# limit=1 isolates the primary hit to the single best vector match, so the
# second record can only appear via graph expansion — the deterministic hashing
# embedder otherwise returns every record in so small a store.


@pytest.mark.asyncio
async def test_recall_expands_forward_links():
    svc = await _make_service()
    a = await _promote_active(svc, "alpha apples orchard cider")
    b = await _promote_active(svc, "beta bananas plantation tropical", links=[a])

    # The query matches B by content; A is pulled in via B's forward edge.
    results = await svc.recall(query="beta bananas plantation tropical", limit=1)
    ids = [r.memory_id for r in results]
    assert ids[0] == b
    assert a in ids


@pytest.mark.asyncio
async def test_recall_expands_reverse_links():
    svc = await _make_service()
    a = await _promote_active(svc, "alpha apples orchard cider")
    b = await _promote_active(svc, "beta bananas plantation tropical", links=[a])

    # The query matches A; B is pulled in because it links back to A.
    results = await svc.recall(query="alpha apples orchard cider", limit=1)
    ids = [r.memory_id for r in results]
    assert ids[0] == a
    assert b in ids


@pytest.mark.asyncio
async def test_recall_neighbors_disabled():
    structured = InMemoryStructuredStore().with_scope(AppScope(app_id="app1"))
    vector = FAISSMemoryVectorStore().with_scope(AppScope(app_id="app1"))
    svc = LongTermMemory(
        structured, vector, MagicMock(), HashingEmbedding(),
        app_id="app1", recall_neighbors=0,
    )
    await svc.setup()
    a = await _promote_active(svc, "alpha apples orchard cider")
    await _promote_active(svc, "beta bananas plantation tropical", links=[a])

    results = await svc.recall(query="alpha apples orchard cider", limit=1)
    assert [r.memory_id for r in results] == [a]


@pytest.mark.asyncio
async def test_recall_skips_superseded_neighbor():
    svc = await _make_service()
    a = await _promote_active(svc, "alpha apples orchard cider")
    b = await _promote_active(svc, "beta bananas plantation tropical", links=[a])
    # Supersede the forward-linked neighbor; the dangling edge is skipped.
    recs = {r.memory_id: r for r in await svc._load_records([a])}
    recs[a].status = MemoryStatus.SUPERSEDED
    await svc._save_record(recs[a])

    results = await svc.recall(query="beta bananas plantation tropical", limit=1)
    ids = [r.memory_id for r in results]
    assert ids == [b]


# ---------------------------------------------------------------------------
# reconcile domain guidance (the consolidation-side analog of distill's
# domain_fact_guidance / mem0's custom_update_memory_prompt)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconcile_guidance_injected_into_decide_prompt():
    structured = InMemoryStructuredStore().with_scope(AppScope(app_id="app1"))
    vector = FAISSMemoryVectorStore().with_scope(AppScope(app_id="app1"))
    llm = _llm_returning({"operation": "NOOP"})
    svc = LongTermMemory(
        structured, vector, llm, HashingEmbedding(),
        app_id="app1",
        reconcile_guidance="Clauses with different effective dates are distinct.",
    )
    await svc.setup()
    # Seed a related active record (shared entity) so reconcile reaches _decide.
    await svc.promote(
        candidate=_candidate("contract clause on liability", entities=("acme",)),
        status=MemoryStatus.ACTIVE,
    )
    await svc.reconcile(
        candidate=_candidate("contract clause on liability cap", entities=("acme",)),
    )

    system_prompt = llm.complete.call_args.args[0][0]["content"]
    assert "Domain reconciliation guidance" in system_prompt
    assert "different effective dates are distinct" in system_prompt
    # The guidance is additive — it sits above the operation rules, not replacing them.
    assert system_prompt.index("Domain reconciliation guidance") < system_prompt.index("Rules:")


@pytest.mark.asyncio
async def test_reconcile_without_guidance_uses_generic_prompt():
    svc = await _make_service(llm=_llm_returning({"operation": "NOOP"}))
    await svc.promote(
        candidate=_candidate("contract clause on liability", entities=("acme",)),
        status=MemoryStatus.ACTIVE,
    )
    await svc.reconcile(
        candidate=_candidate("contract clause on liability cap", entities=("acme",)),
    )
    system_prompt = svc._llm.complete.call_args.args[0][0]["content"]
    assert "Domain reconciliation guidance" not in system_prompt
