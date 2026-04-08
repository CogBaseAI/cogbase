"""Contract tests for StructuredStoreBase — run against every concrete adapter."""

import pytest

from cogbase.core.models import Contradiction, Event, Fact


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def make_fact(**kwargs) -> Fact:
    defaults = dict(
        type="notice_period",
        value="60 days",
        raw_text="sixty (60) days written notice",
        doc_id="doc-1",
        confidence=0.95,
    )
    return Fact(**{**defaults, **kwargs})


def make_event(**kwargs) -> Event:
    defaults = dict(session_id="session-1", actor="user", action="query")
    return Event(**{**defaults, **kwargs})


def make_contradiction(fact_a: Fact, fact_b: Fact, **kwargs) -> Contradiction:
    defaults = dict(conflict_type="date")
    return Contradiction(fact_a=fact_a, fact_b=fact_b, **{**defaults, **kwargs})


# ------------------------------------------------------------------
# Facts
# ------------------------------------------------------------------

def test_save_and_query_facts(structured_store):
    fact = make_fact()
    structured_store.save_facts([fact])
    results = structured_store.query_facts({})
    assert len(results) == 1
    assert results[0].fact_id == fact.fact_id


def test_query_facts_filter_by_type(structured_store):
    structured_store.save_facts([
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
    ])
    results = structured_store.query_facts({"type": "notice_period"})
    assert len(results) == 1
    assert results[0].type == "notice_period"


def test_query_facts_filter_by_doc_id(structured_store):
    structured_store.save_facts([
        make_fact(doc_id="doc-1"),
        make_fact(doc_id="doc-2"),
    ])
    results = structured_store.query_facts({"doc_id": "doc-1"})
    assert all(f.doc_id == "doc-1" for f in results)
    assert len(results) == 1


def test_query_facts_multi_filter(structured_store):
    structured_store.save_facts([
        make_fact(type="notice_period", doc_id="doc-1"),
        make_fact(type="notice_period", doc_id="doc-2"),
        make_fact(type="termination_date", doc_id="doc-1"),
    ])
    results = structured_store.query_facts({"type": "notice_period", "doc_id": "doc-1"})
    assert len(results) == 1


def test_query_facts_empty_filters_returns_all(structured_store):
    structured_store.save_facts([make_fact(), make_fact(), make_fact()])
    assert len(structured_store.query_facts({})) == 3


def test_query_facts_no_match_returns_empty(structured_store):
    structured_store.save_facts([make_fact(type="notice_period")])
    assert structured_store.query_facts({"type": "nonexistent"}) == []


def test_facts_upsert_by_id(structured_store):
    fact = make_fact(value="30 days")
    structured_store.save_facts([fact])
    updated = Fact(
        fact_id=fact.fact_id,
        type=fact.type,
        value="60 days",
        raw_text=fact.raw_text,
        doc_id=fact.doc_id,
        confidence=fact.confidence,
    )
    structured_store.save_facts([updated])
    results = structured_store.query_facts({"doc_id": fact.doc_id})
    assert len(results) == 1
    assert results[0].value == "60 days"


# ------------------------------------------------------------------
# Timeline
# ------------------------------------------------------------------

def test_save_and_query_timeline(structured_store):
    events = [make_event(), make_event()]
    structured_store.save_timeline(events)
    results = structured_store.query_timeline("session-1")
    assert len(results) == 2


def test_query_timeline_isolates_by_session(structured_store):
    structured_store.save_timeline([
        make_event(session_id="session-1"),
        make_event(session_id="session-2"),
        make_event(session_id="session-1"),
    ])
    assert len(structured_store.query_timeline("session-1")) == 2
    assert len(structured_store.query_timeline("session-2")) == 1


def test_query_timeline_ordered_by_timestamp(structured_store):
    from datetime import datetime, timezone

    t1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

    structured_store.save_timeline([
        make_event(session_id="s", timestamp=t1, action="first"),
        make_event(session_id="s", timestamp=t2, action="third"),
        make_event(session_id="s", timestamp=t3, action="second"),
    ])
    results = structured_store.query_timeline("s")
    assert [r.action for r in results] == ["first", "second", "third"]


def test_query_timeline_unknown_session_returns_empty(structured_store):
    assert structured_store.query_timeline("no-such-session") == []


def test_timeline_event_payload_roundtrip(structured_store):
    event = make_event(payload={"query": "what is the notice period?", "count": 3})
    structured_store.save_timeline([event])
    results = structured_store.query_timeline(event.session_id)
    assert results[0].payload == {"query": "what is the notice period?", "count": 3}


# ------------------------------------------------------------------
# Contradictions
# ------------------------------------------------------------------

def test_save_and_query_contradiction(structured_store):
    fa, fb = make_fact(doc_id="doc-1"), make_fact(doc_id="doc-2")
    c = make_contradiction(fa, fb)
    structured_store.save_contradiction(c)
    results = structured_store.query_contradictions({})
    assert len(results) == 1
    assert results[0].contradiction_id == c.contradiction_id


def test_contradiction_nested_facts_roundtrip(structured_store):
    fa = make_fact(doc_id="doc-1", value="30 days")
    fb = make_fact(doc_id="doc-2", value="60 days")
    c = make_contradiction(fa, fb, conflict_type="numeric")
    structured_store.save_contradiction(c)
    results = structured_store.query_contradictions({})
    assert results[0].fact_a.value == "30 days"
    assert results[0].fact_b.value == "60 days"


def test_query_contradictions_filter_by_type(structured_store):
    fa, fb = make_fact(), make_fact()
    structured_store.save_contradiction(make_contradiction(fa, fb, conflict_type="date"))
    structured_store.save_contradiction(make_contradiction(fa, fb, conflict_type="numeric"))
    results = structured_store.query_contradictions({"conflict_type": "date"})
    assert len(results) == 1
    assert results[0].conflict_type == "date"


def test_query_contradictions_filter_by_resolved(structured_store):
    fa, fb = make_fact(), make_fact()
    structured_store.save_contradiction(make_contradiction(fa, fb, resolved=False))
    structured_store.save_contradiction(make_contradiction(fa, fb, resolved=True))
    assert len(structured_store.query_contradictions({"resolved": False})) == 1
    assert len(structured_store.query_contradictions({"resolved": True})) == 1


def test_query_contradictions_filter_by_doc_id(structured_store):
    fa = make_fact(doc_id="doc-1")
    fb = make_fact(doc_id="doc-2")
    fc = make_fact(doc_id="doc-3")
    structured_store.save_contradiction(make_contradiction(fa, fb))
    structured_store.save_contradiction(make_contradiction(fb, fc))
    # doc-2 appears in both contradictions
    results = structured_store.query_contradictions({"doc_id": "doc-2"})
    assert len(results) == 2
    # doc-1 appears in only one
    results = structured_store.query_contradictions({"doc_id": "doc-1"})
    assert len(results) == 1


def test_contradiction_upsert_by_id(structured_store):
    fa, fb = make_fact(), make_fact()
    c = make_contradiction(fa, fb, resolved=False)
    structured_store.save_contradiction(c)
    resolved = Contradiction(
        contradiction_id=c.contradiction_id,
        fact_a=fa,
        fact_b=fb,
        conflict_type=c.conflict_type,
        resolved=True,
        resolution_note="confirmed same clause",
    )
    structured_store.save_contradiction(resolved)
    results = structured_store.query_contradictions({})
    assert len(results) == 1
    assert results[0].resolved is True
    assert results[0].resolution_note == "confirmed same clause"
