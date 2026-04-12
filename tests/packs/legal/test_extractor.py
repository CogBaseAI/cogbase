"""Tests for ClauseExtractor."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from packs.legal.extractor import ClauseExtractor
from packs.legal.schema import CLAUSES_COLLECTION, CLAUSES_SCHEMA, Clause


def _make_client(content: str) -> MagicMock:
    """Build a minimal mock OpenAI client that returns *content*."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    response = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Schema / collection properties
# ---------------------------------------------------------------------------

def test_collection_name():
    extractor = ClauseExtractor(MagicMock(), model="test-model")
    assert extractor.collection == CLAUSES_COLLECTION


def test_schema_returned():
    extractor = ClauseExtractor(MagicMock(), model="test-model")
    assert extractor.schema == CLAUSES_SCHEMA


# ---------------------------------------------------------------------------
# extract() — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_clause_records():
    payload = json.dumps([
        {"type": "payment", "text": "Payment is due within 30 days.", "confidence": 0.95},
        {"type": "termination", "text": "Either party may terminate with 60 days notice.", "confidence": 0.9},
    ])
    extractor = ClauseExtractor(_make_client(payload), model="test-model")
    results = await extractor.extract("some contract text", doc_id="doc-001")

    assert len(results) == 2
    assert all(isinstance(r, Clause) for r in results)

    types = {r.type for r in results}
    assert types == {"payment", "termination"}


@pytest.mark.asyncio
async def test_extract_clause_id_contains_doc_id():
    payload = json.dumps([
        {"type": "liability", "text": "Liability is capped at $1M.", "confidence": 0.88},
    ])
    extractor = ClauseExtractor(_make_client(payload), model="test-model")
    results = await extractor.extract("contract text", doc_id="contract-42")

    assert results[0].clause_id.startswith("contract-42_liability_")


@pytest.mark.asyncio
async def test_extract_preserves_verbatim_text():
    verbatim = "Governing law shall be the laws of the State of New York."
    payload = json.dumps([
        {"type": "governing_law", "text": verbatim, "confidence": 0.99},
    ])
    extractor = ClauseExtractor(_make_client(payload), model="test-model")
    results = await extractor.extract("contract text", doc_id="doc-002")

    assert results[0].text == verbatim


@pytest.mark.asyncio
async def test_confidence_clamped_to_valid_range():
    payload = json.dumps([
        {"type": "other", "text": "Some clause.", "confidence": 1.5},
        {"type": "other", "text": "Another clause.", "confidence": -0.2},
    ])
    extractor = ClauseExtractor(_make_client(payload), model="test-model")
    results = await extractor.extract("contract text", doc_id="doc-003")

    for r in results:
        assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# extract() — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_empty_text_returns_empty():
    extractor = ClauseExtractor(_make_client("[]"), model="test-model")
    results = await extractor.extract("   ", doc_id="doc-004")
    assert results == []


@pytest.mark.asyncio
async def test_extract_llm_returns_empty_array():
    extractor = ClauseExtractor(_make_client("[]"), model="test-model")
    results = await extractor.extract("contract text", doc_id="doc-005")
    assert results == []


@pytest.mark.asyncio
async def test_extract_invalid_json_returns_empty():
    extractor = ClauseExtractor(_make_client("not json at all"), model="test-model")
    results = await extractor.extract("contract text", doc_id="doc-006")
    assert results == []


@pytest.mark.asyncio
async def test_extract_non_array_json_returns_empty():
    extractor = ClauseExtractor(_make_client('{"type": "payment"}'), model="test-model")
    results = await extractor.extract("contract text", doc_id="doc-007")
    assert results == []


@pytest.mark.asyncio
async def test_extract_missing_confidence_uses_default():
    payload = json.dumps([
        {"type": "notice", "text": "30 days written notice required."},
    ])
    extractor = ClauseExtractor(_make_client(payload), model="test-model")
    results = await extractor.extract("contract text", doc_id="doc-008")

    assert results[0].confidence == 0.8


@pytest.mark.asyncio
async def test_extract_unique_ids_for_same_type():
    payload = json.dumps([
        {"type": "payment", "text": "First payment clause.", "confidence": 0.9},
        {"type": "payment", "text": "Second payment clause.", "confidence": 0.85},
    ])
    extractor = ClauseExtractor(_make_client(payload), model="test-model")
    results = await extractor.extract("contract text", doc_id="doc-009")

    ids = [r.clause_id for r in results]
    assert len(ids) == len(set(ids)), "clause_ids must be unique"
