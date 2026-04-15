"""Tests for ContractExtractor."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.models import Document
from packs.legal.extractor import ContractExtractor
from packs.legal.schema import CONTRACTS_COLLECTION, CONTRACTS_SCHEMA, ContractRecord, Party, PaymentTerms


def _make_client(content: str) -> MagicMock:
    """Build a minimal mock OpenAI client that returns *content*."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    response = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _full_payload(**overrides) -> str:
    """Return a complete valid LLM response JSON string."""
    data = {
        "contract_type": "NDA",
        "purpose": "Mutual non-disclosure between two technology companies.",
        "effective_date": "2024-03-01",
        "expiry_date": "2026-03-01",
        "parties": [
            {"name": "Acme Corp", "role": "discloser", "jurisdiction": None},
            {"name": "Supplier Ltd", "role": "recipient", "jurisdiction": None},
        ],
        "contract_value": None,
        "currency": None,
        "payment_terms": None,
        "termination": "Either party may terminate with 30 days written notice.",
        "liability": "Neither party shall be liable for indirect damages.",
        "governing_law": "This agreement is governed by the laws of England and Wales.",
        "confidentiality": "Each party shall keep the other's information strictly confidential.",
        "indemnification": None,
        "dispute_resolution": None,
        "notice_period_days": 30,
        "liability_cap": None,
        "key_terms": ["Confidential Information: Any non-public data shared between parties."],
        "special_conditions": [],
    }
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Schema / collection properties
# ---------------------------------------------------------------------------

def test_collection_name():
    extractor = ContractExtractor(MagicMock(), model="test-model")
    assert extractor.collection == CONTRACTS_COLLECTION


def test_schema_returned():
    extractor = ContractExtractor(MagicMock(), model="test-model")
    assert extractor.schema == CONTRACTS_SCHEMA


# ---------------------------------------------------------------------------
# extract() — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_one_contract_record():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert isinstance(result, ContractRecord)


@pytest.mark.asyncio
async def test_extract_contract_id_contains_doc_id():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    result = await extractor.extract(Document(doc_id="vendor-42", text="contract text"))

    assert result.contract_id.startswith("vendor-42_")


@pytest.mark.asyncio
async def test_extract_doc_id_set():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert result.doc_id == "doc-001"


@pytest.mark.asyncio
async def test_extract_contract_basics():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert r.contract_type == "NDA"
    assert r.effective_date == "2024-03-01"
    assert r.expiry_date == "2026-03-01"
    assert len(r.parties) == 2
    assert r.parties[0].name == "Acme Corp"
    assert r.parties[0].role == "discloser"
    assert r.parties[1].name == "Supplier Ltd"


@pytest.mark.asyncio
async def test_extract_common_clause_text_verbatim():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert r.termination == "Either party may terminate with 30 days written notice."
    assert r.governing_law == "This agreement is governed by the laws of England and Wales."
    assert r.confidentiality == "Each party shall keep the other's information strictly confidential."


@pytest.mark.asyncio
async def test_extract_absent_clauses_are_none():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert r.payment_terms is None
    assert r.indemnification is None
    assert r.dispute_resolution is None
    assert r.contract_value is None
    assert r.liability_cap is None


@pytest.mark.asyncio
async def test_extract_notice_period_days():
    extractor = ContractExtractor(_make_client(_full_payload(notice_period_days=30)), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert r.notice_period_days == 30


@pytest.mark.asyncio
async def test_extract_key_terms():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert len(r.key_terms) == 1
    assert isinstance(r.key_terms[0], str)
    assert "Confidential Information" in r.key_terms[0]


@pytest.mark.asyncio
async def test_extract_special_conditions():
    payload = _full_payload(special_conditions=[
        "This agreement supersedes all prior NDAs between the parties.",
        "Obligations survive termination for 5 years.",
    ])
    extractor = ContractExtractor(_make_client(payload), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert len(r.special_conditions) == 2
    assert "supersedes" in r.special_conditions[0]


@pytest.mark.asyncio
async def test_extract_contract_value_and_currency():
    payload = _full_payload(contract_value=250000.0, currency="USD")
    extractor = ContractExtractor(_make_client(payload), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert r.contract_value == 250000.0
    assert r.currency == "USD"


@pytest.mark.asyncio
async def test_extract_liability_cap():
    payload = _full_payload(liability_cap=500000.0, currency="GBP")
    extractor = ContractExtractor(_make_client(payload), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert r.liability_cap == 500000.0


# ---------------------------------------------------------------------------
# extract() — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_empty_text_returns_none():
    extractor = ContractExtractor(_make_client("{}"), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-002", text="   "))
    assert result is None


@pytest.mark.asyncio
async def test_extract_invalid_json_returns_none():
    extractor = ContractExtractor(_make_client("not json"), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-003", text="contract text"))
    assert result is None


@pytest.mark.asyncio
async def test_extract_json_array_instead_of_object_returns_none():
    """LLM accidentally returns an array — should return None, not crash."""
    extractor = ContractExtractor(_make_client("[1, 2, 3]"), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-004", text="contract text"))
    assert result is None


@pytest.mark.asyncio
async def test_extract_missing_fields_default_to_none_or_empty():
    """Minimal response — only the outer object, no fields."""
    extractor = ContractExtractor(_make_client("{}"), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-005", text="contract text"))

    assert isinstance(result, ContractRecord)
    assert result.contract_type is None
    assert result.parties == []
    assert result.key_terms == []
    assert result.special_conditions == []


@pytest.mark.asyncio
async def test_extract_invalid_numeric_fields_rejects_record():
    """Non-numeric strings for int/float fields fail Pydantic validation — record is None."""
    payload = _full_payload(notice_period_days="thirty", contract_value="one million")
    extractor = ContractExtractor(_make_client(payload), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-006", text="contract text"))

    assert result is None


@pytest.mark.asyncio
async def test_extract_explicit_null_fields():
    payload = _full_payload(parties=[], effective_date=None)
    extractor = ContractExtractor(_make_client(payload), model="test-model")
    r = await extractor.extract(Document(doc_id="doc-007", text="contract text"))

    assert r.parties == []
    assert r.effective_date is None


@pytest.mark.asyncio
async def test_extract_non_list_key_terms_rejects_record():
    """A scalar where a list is expected fails Pydantic validation — record is None."""
    payload = _full_payload(key_terms="not a list")
    extractor = ContractExtractor(_make_client(payload), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-008", text="contract text"))

    assert result is None


@pytest.mark.asyncio
async def test_extract_non_list_special_conditions_rejects_record():
    """A dict where a list is expected fails Pydantic validation — record is None."""
    payload = _full_payload(special_conditions={"key": "value"})
    extractor = ContractExtractor(_make_client(payload), model="test-model")
    result = await extractor.extract(Document(doc_id="doc-009", text="contract text"))

    assert result is None


@pytest.mark.asyncio
async def test_extract_unique_contract_ids_per_call():
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model")
    r1 = await extractor.extract(Document(doc_id="doc-010", text="text"))
    r2 = await extractor.extract(Document(doc_id="doc-010", text="text"))
    assert r1.contract_id != r2.contract_id


# ---------------------------------------------------------------------------
# extract() — retry behaviour
# ---------------------------------------------------------------------------

def _make_client_with_responses(*contents: str) -> MagicMock:
    """Build a mock client that returns each content string in sequence."""
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=c))])
        for c in contents
    ]
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=responses)
    return client


@pytest.mark.asyncio
async def test_extract_succeeds_on_retry_after_bad_json(monkeypatch):
    """First call returns bad JSON; second call returns valid JSON — should succeed."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    client = _make_client_with_responses("not json", _full_payload())
    extractor = ContractExtractor(client, model="test-model", max_retries=2)
    result = await extractor.extract(Document(doc_id="doc-retry-1", text="contract text"))

    assert isinstance(result, ContractRecord)
    assert client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_extract_returns_none_after_all_retries_exhausted(monkeypatch):
    """All attempts return bad JSON — should return None."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    client = _make_client_with_responses("bad", "bad", "bad")
    extractor = ContractExtractor(client, model="test-model", max_retries=2)
    result = await extractor.extract(Document(doc_id="doc-retry-2", text="contract text"))

    assert result is None
    assert client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_extract_no_retry_on_success(monkeypatch):
    """First call returns valid JSON — should not retry."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    extractor = ContractExtractor(_make_client(_full_payload()), model="test-model", max_retries=2)
    result = await extractor.extract(Document(doc_id="doc-retry-3", text="contract text"))

    assert isinstance(result, ContractRecord)
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_extract_retry_uses_exponential_backoff(monkeypatch):
    """Sleep is called with 1s then 2s for two retries."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    client = _make_client_with_responses("bad", "bad", _full_payload())
    extractor = ContractExtractor(client, model="test-model", max_retries=2)
    await extractor.extract(Document(doc_id="doc-retry-4", text="contract text"))

    assert sleep_mock.call_count == 2
    assert sleep_mock.call_args_list[0].args[0] == 1   # 2^0
    assert sleep_mock.call_args_list[1].args[0] == 2   # 2^1


@pytest.mark.asyncio
async def test_extract_max_retries_zero_no_sleep(monkeypatch):
    """max_retries=0 means one attempt only; no sleep on failure."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    extractor = ContractExtractor(_make_client("bad json"), model="test-model", max_retries=0)
    result = await extractor.extract(Document(doc_id="doc-retry-5", text="contract text"))

    assert result is None
    sleep_mock.assert_not_called()
