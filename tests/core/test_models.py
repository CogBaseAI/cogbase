import pytest
from datetime import datetime
from pydantic import ValidationError

from cogbase.core.models import Chunk, Contradiction, Event, Fact


def make_fact(**kwargs):
    defaults = dict(type="date", value="2024-01-01", raw_text="Jan 1, 2024", doc_id="doc1", confidence=0.9)
    return Fact(**{**defaults, **kwargs})


def test_fact_defaults():
    f = make_fact()
    assert isinstance(f.fact_id, str) and f.fact_id
    assert isinstance(f.confidence, float)


def test_fact_confidence_validation():
    with pytest.raises(ValidationError):
        make_fact(confidence=1.5)


def test_fact_is_frozen():
    f = make_fact()
    with pytest.raises(Exception):
        f.value = "new"  # type: ignore[misc]


def test_chunk_embedding_optional():
    c = Chunk(chunk_id="doc1_0", doc_id="doc1", text="hello")
    assert c.embedding is None


def test_contradiction_links_facts():
    fa = make_fact(doc_id="doc1")
    fb = make_fact(doc_id="doc2")
    c = Contradiction(fact_a=fa, fact_b=fb, conflict_type="date")
    assert c.fact_a.doc_id == "doc1"
    assert c.fact_b.doc_id == "doc2"


def test_event_timestamp_default():
    e = Event(session_id="s1", actor="user", action="query")
    assert isinstance(e.timestamp, datetime)
