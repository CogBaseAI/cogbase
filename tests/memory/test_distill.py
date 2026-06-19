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
         "source_seqs": [0], "confidence": 0.8},
        {"content": "user works at Acme Corp", "kind": "fact",
         "source_seqs": [2], "confidence": 0.7},
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
         "entities": ["Acme Corp", "acme corp"], "source_seqs": [0],
         "confidence": 0.7},
    ])
    distiller = Distiller(episodic, lt, llm)

    ids = await distiller.distill_session(session_id=sid)
    rec = (await lt._load_records(ids))[0]
    assert rec.entities == ["acme corp"]


@pytest.mark.asyncio
async def test_distill_carries_llm_confidence(episodic):
    sid = "sess-conf"
    await _seed_turn(episodic, sid, "I always want concise answers", "Noted.")
    await _seed_turn(episodic, sid, "I think I might use React", "OK.")
    lt = await _long_term()
    llm = _extracting_llm([
        # Explicit confidence is carried through verbatim.
        {"content": "user prefers concise answers", "kind": "preference",
         "source_seqs": [0], "confidence": 0.95},
        # Out-of-range is clamped to [0, 1].
        {"content": "user might use React", "kind": "preference",
         "source_seqs": [2], "confidence": 1.5},
    ])
    distiller = Distiller(episodic, lt, llm)

    ids = await distiller.distill_session(session_id=sid)
    recs = {r.content: r for r in await lt._load_records(ids)}
    assert recs["user prefers concise answers"].confidence == 0.95
    assert recs["user might use React"].confidence == 1.0


@pytest.mark.asyncio
async def test_distill_abandons_candidates_below_their_kind_floor(episodic):
    sid = "sess-low"
    await _seed_turn(episodic, sid, "I always want concise answers", "Noted.")
    await _seed_turn(episodic, sid, "you keep mixing up Berlin and Munich", "Sorry.")
    lt = await _long_term()
    # Same 0.65 score, two kinds: it clears the retrieval_hint floor (0.6) but
    # falls below the higher-stakes correction floor (0.7), so only the hint is
    # distilled — the floor is per-kind, not flat.
    llm = _extracting_llm([
        {"content": "route Berlin/Munich questions to the geography tool",
         "kind": "retrieval_hint", "source_seqs": [0], "confidence": 0.65},
        {"content": "user is based in Munich", "kind": "correction",
         "source_seqs": [2], "confidence": 0.65},
    ])
    distiller = Distiller(episodic, lt, llm)

    ids = await distiller.distill_session(session_id=sid)
    recs = [r.content for r in await lt._load_records(ids)]
    assert recs == ["route Berlin/Munich questions to the geography tool"]


@pytest.mark.asyncio
async def test_distill_forces_confidence_rejecting_responses_without_it(episodic):
    sid = "sess-noconf"
    await _seed_turn(episodic, sid, "I always want concise answers", "Noted.")
    lt = await _long_term()
    # confidence is a required field of the extraction schema: a response that
    # omits it fails validation, so after retries nothing is distilled rather
    # than silently falling back to a default score.
    llm = _extracting_llm([
        {"content": "user prefers concise answers", "kind": "preference",
         "source_seqs": [0]},
    ])
    distiller = Distiller(episodic, lt, llm)

    assert await distiller.distill_session(session_id=sid) == []


@pytest.mark.asyncio
async def test_distill_empty_thread_returns_nothing(episodic):
    lt = await _long_term()
    distiller = Distiller(episodic, lt, _extracting_llm([]))
    assert await distiller.distill_session(session_id="empty") == []


@pytest.mark.asyncio
async def test_distill_anchors_transcript_to_session_observation_date(episodic):
    # The conversation happened when the log was written; distillation runs
    # offline, so the prompt must anchor relative time refs ("yesterday") to the
    # session date — not to whenever distill happens to run.
    sid = "sess-temporal"
    await _seed_turn(episodic, sid, "I met the investor yesterday", "Noted.")
    events = await episodic.replay(session_id=sid)
    session_date = min(e.created_at for e in events)

    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "user met the investor", "kind": "fact",
         "source_seqs": [0], "confidence": 0.8},
    ])
    distiller = Distiller(episodic, lt, llm)
    await distiller.distill_session(session_id=sid)

    # The user message carries the session's observation date as the anchor.
    user_msg = llm.complete.call_args.args[0][1]["content"]
    assert f"{session_date:%Y-%m-%d}" in user_msg
    assert "Observation date" in user_msg


@pytest.mark.asyncio
async def test_distill_drops_candidate_with_bad_kind(episodic):
    sid = "sess-2"
    await _seed_turn(episodic, sid, "hello", "hi")
    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "valid one", "kind": "preference", "source_seqs": [0],
         "confidence": 0.7},
        {"content": "broken", "kind": "nonsense", "source_seqs": [0],
         "confidence": 0.7},
    ])
    # The bad-kind item is schema-invalid, so extraction fails validation and
    # nothing is returned — verifies we never promote a malformed candidate.
    distiller = Distiller(episodic, lt, llm)
    ids = await distiller.distill_session(session_id=sid)
    assert ids == []


@pytest.mark.asyncio
async def test_distill_injects_existing_memories_for_dedup(episodic):
    # Front-loaded existing memories appear in the extraction prompt as a dedup
    # reference so the extractor can skip already-captured claims, instead of the
    # duplicate only being caught later in reconcile.
    lt = await _long_term()
    # Seed an active memory the recall will surface.
    from cogbase.memory.models import MemoryCandidate, MemoryKind

    await lt.promote(
        candidate=MemoryCandidate(
            content="user prefers concise answers",
            kind=MemoryKind.PREFERENCE,
            confidence=0.9,
        )
    )

    sid = "sess-existing"
    await _seed_turn(episodic, sid, "remember I like concise answers", "Noted.")

    llm = _extracting_llm([])
    distiller = Distiller(episodic, lt, llm)
    await distiller.distill_session(session_id=sid)

    user_msg = llm.complete.call_args.args[0][1]["content"]
    assert "## Existing memories" in user_msg
    assert "user prefers concise answers" in user_msg


@pytest.mark.asyncio
async def test_distill_no_existing_memory_block_when_disabled(episodic):
    lt = await _long_term()
    from cogbase.memory.models import MemoryCandidate, MemoryKind

    await lt.promote(
        candidate=MemoryCandidate(
            content="user prefers concise answers",
            kind=MemoryKind.PREFERENCE,
            confidence=0.9,
        )
    )
    sid = "sess-disabled"
    await _seed_turn(episodic, sid, "remember I like concise answers", "Noted.")

    llm = _extracting_llm([])
    distiller = Distiller(episodic, lt, llm, existing_memory_limit=0)
    await distiller.distill_session(session_id=sid)

    user_msg = llm.complete.call_args.args[0][1]["content"]
    assert "## Existing memories" not in user_msg


@pytest.mark.asyncio
async def test_distill_links_new_memory_to_existing(episodic):
    # The extractor references the masked id from the existing-memories block;
    # the distiller resolves it to the real memory_id and stores the edge.
    from cogbase.memory.models import MemoryCandidate, MemoryKind

    lt = await _long_term()
    target_id = await lt.promote(
        candidate=MemoryCandidate(
            content="user has a dog named Max",
            kind=MemoryKind.FACT,
            confidence=0.9,
            entities=["max"],
        )
    )

    sid = "sess-link"
    await _seed_turn(episodic, sid, "Max and I went camping and hiked", "Nice!")

    llm = _extracting_llm([
        {"content": "user went camping with Max and hiked", "kind": "fact",
         "source_seqs": [0], "confidence": 0.8, "entities": ["max"],
         "linked_memory_ids": [0]},
    ])
    distiller = Distiller(episodic, lt, llm)
    ids = await distiller.distill_session(session_id=sid)

    # The existing memory's id was shown as [id=0] in the prompt.
    user_msg = llm.complete.call_args.args[0][1]["content"]
    assert "[id=0] user has a dog named Max" in user_msg

    new = (await lt._load_records(ids))[0]
    assert new.linked_memory_ids == [target_id]


@pytest.mark.asyncio
async def test_distill_is_idempotent_across_reruns(episodic):
    # Sessions are resumable / re-closable, so distillation can run more than once
    # over the same log.  A session_distilled watermark records how far it has
    # extracted, so a re-run over an unchanged log re-extracts nothing and does
    # not re-reconcile — which would otherwise reinforce the record and inflate
    # its confidence with no new evidence.
    sid = "sess-rerun"
    await _seed_turn(episodic, sid, "I work at Acme Corp", "Got it.")
    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "user works at Acme Corp", "kind": "fact",
         "source_seqs": [0], "confidence": 0.9},
    ])
    distiller = Distiller(episodic, lt, llm)

    ids = await distiller.distill_session(session_id=sid)
    assert len(ids) == 1
    first_conf = (await lt._load_records(ids))[0].confidence
    calls_after_first = llm.complete.call_count

    # Re-distilling the unchanged log finds no turns past the watermark.
    assert await distiller.distill_session(session_id=sid) == []
    # The extraction LLM is not called again (the thread past the watermark is empty).
    assert llm.complete.call_count == calls_after_first
    # Confidence was not re-inflated by a re-reconcile.
    assert (await lt._load_records(ids))[0].confidence == first_conf


@pytest.mark.asyncio
async def test_distill_resumed_session_extracts_only_new_turns(episodic):
    # A resumed session is re-distilled: only turns appended past the watermark
    # reach the extractor, not the already-distilled transcript.
    sid = "sess-resume"
    await _seed_turn(episodic, sid, "I work at Acme Corp", "Got it.")
    lt = await _long_term()
    llm = _extracting_llm([
        {"content": "user works at Acme Corp", "kind": "fact",
         "source_seqs": [0], "confidence": 0.9},
    ])
    distiller = Distiller(episodic, lt, llm)
    await distiller.distill_session(session_id=sid)

    # Resume: a new turn (user seq 3, after the watermark event at seq 2) arrives.
    await _seed_turn(episodic, sid, "I drive a Ferrari 488 GTB", "Nice.")
    llm.complete = AsyncMock(return_value={"content": json.dumps({"memories": [
        {"content": "user drives a Ferrari 488 GTB", "kind": "fact",
         "source_seqs": [3], "confidence": 0.9},
    ]})})
    await distiller.distill_session(session_id=sid)

    # Only the new turn is in the second extraction's transcript; the already-
    # distilled first turn (its transcript line) is gone.
    user_msg = llm.complete.call_args.args[0][1]["content"]
    assert "[3 user] I drive a Ferrari 488 GTB" in user_msg
    assert "[0 user]" not in user_msg


@pytest.mark.asyncio
async def test_distill_advances_watermark_when_nothing_extracted(episodic):
    # A successful extraction that yields no memories still advances the watermark:
    # those turns were judged (chit-chat) and must not be re-examined next pass.
    sid = "sess-empty-extract"
    await _seed_turn(episodic, sid, "lol nice weather", "Indeed!")
    lt = await _long_term()
    llm = _extracting_llm([])
    distiller = Distiller(episodic, lt, llm)

    assert await distiller.distill_session(session_id=sid) == []
    events = await episodic.replay(session_id=sid)
    from cogbase.memory.projection import latest_distillation
    assert latest_distillation(events) == 1  # the final_answer turn's seq

    # Re-distilling finds nothing new and never calls the extractor again.
    calls = llm.complete.call_count
    assert await distiller.distill_session(session_id=sid) == []
    assert llm.complete.call_count == calls


@pytest.mark.asyncio
async def test_distill_does_not_watermark_on_extraction_failure(episodic):
    # An unparseable extraction (failure, not "nothing to extract") leaves the
    # turns un-watermarked so a later pass retries them rather than silently
    # skipping them forever.
    sid = "sess-fail"
    await _seed_turn(episodic, sid, "I work at Acme Corp", "Got it.")
    lt = await _long_term()
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": "not json at all"})
    distiller = Distiller(episodic, lt, llm)

    assert await distiller.distill_session(session_id=sid) == []
    events = await episodic.replay(session_id=sid)
    from cogbase.memory.projection import latest_distillation
    assert latest_distillation(events) == -1  # no watermark written


@pytest.mark.asyncio
async def test_distill_drops_unresolvable_link_id(episodic):
    # A link id the extractor never saw (out of range / hallucinated) degrades to
    # no edge rather than a dangling reference.
    sid = "sess-badlink"
    await _seed_turn(episodic, sid, "I work at Acme", "Got it.")
    lt = await _long_term()  # empty store -> no existing memories, no valid ids
    llm = _extracting_llm([
        {"content": "user works at Acme", "kind": "fact", "source_seqs": [0],
         "confidence": 0.8, "linked_memory_ids": [3]},
    ])
    distiller = Distiller(episodic, lt, llm)
    ids = await distiller.distill_session(session_id=sid)
    assert (await lt._load_records(ids))[0].linked_memory_ids == []
