"""Tests for LLMExtractor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from cogbase.config.config import ExtractorConfig
from cogbase.llms import LLMBase
from cogbase.llms.compaction import estimate_tokens
from cogbase.core.models import Document
from cogbase.pipeline.chunking.langchain import split_text_by_tokens
from cogbase.pipeline.extraction.llm import LLMExtractor
from examples.contract_analyst_demo.schema import (
    ContractExtraction,
)


_DEFAULT_EXTRACTION_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'


def _make_llm(content: str) -> MagicMock:
    """Build a mock LLMBase returning *content* for complete() and streaming it."""
    llm = MagicMock(spec=LLMBase)
    llm.complete = AsyncMock(return_value={"content": content})
    llm.context_window = MagicMock(return_value=128_000)

    async def _stream(*args, **kwargs):
        yield content

    llm.complete_stream = _stream
    return llm


def _build_record_schema(extraction_schema: dict) -> dict:
    schema = dict(extraction_schema)
    props = dict(schema.get("properties", {}))
    props["doc_id"] = {"type": "string"}
    schema["properties"] = props
    return schema


def _build_list_record_schema(extraction_schema: dict, item_id_field: str) -> dict:
    schema = dict(extraction_schema)
    props = dict(schema.get("properties", {}))
    props["doc_id"] = {"type": "string"}
    props[item_id_field] = {"type": "string"}
    schema["properties"] = props
    return schema


def _make_extractor(llm: MagicMock) -> LLMExtractor:
    extraction_schema = ContractExtraction.model_json_schema()
    return LLMExtractor(
        llm,
        extraction_schema=extraction_schema,
        config=ExtractorConfig(extraction_schema=_DEFAULT_EXTRACTION_SCHEMA, prompt="Extract."),
        record_schema=_build_record_schema(extraction_schema),
    )


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
# extract() — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_one_record():
    extractor = _make_extractor(_make_llm(_full_payload()))
    result = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert result is not None
    assert len(result) == 1
    assert "doc_id" in result[0]


@pytest.mark.asyncio
async def test_extract_doc_id_set():
    extractor = _make_extractor(_make_llm(_full_payload()))
    result = await extractor.extract(Document(doc_id="doc-001", text="contract text"))

    assert result[0]["doc_id"] == "doc-001"


@pytest.mark.asyncio
async def test_extract_contract_basics():
    extractor = _make_extractor(_make_llm(_full_payload()))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert r["contract_type"] == "NDA"
    assert r["effective_date"] == "2024-03-01"
    assert r["expiry_date"] == "2026-03-01"
    assert len(r["parties"]) == 2
    assert r["parties"][0]["name"] == "Acme Corp"
    assert r["parties"][0]["role"] == "discloser"
    assert r["parties"][1]["name"] == "Supplier Ltd"


@pytest.mark.asyncio
async def test_extract_common_clause_text_verbatim():
    extractor = _make_extractor(_make_llm(_full_payload()))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert r["termination"] == "Either party may terminate with 30 days written notice."
    assert r["governing_law"] == "This agreement is governed by the laws of England and Wales."
    assert r["confidentiality"] == "Each party shall keep the other's information strictly confidential."


@pytest.mark.asyncio
async def test_extract_absent_clauses_are_none():
    extractor = _make_extractor(_make_llm(_full_payload()))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert r["payment_terms"] is None
    assert r["indemnification"] is None
    assert r["dispute_resolution"] is None
    assert r["contract_value"] is None
    assert r["liability_cap"] is None


@pytest.mark.asyncio
async def test_extract_notice_period_days():
    extractor = _make_extractor(_make_llm(_full_payload(notice_period_days=30)))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert r["notice_period_days"] == 30


@pytest.mark.asyncio
async def test_extract_key_terms():
    extractor = _make_extractor(_make_llm(_full_payload()))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert len(r["key_terms"]) == 1
    assert isinstance(r["key_terms"][0], str)
    assert "Confidential Information" in r["key_terms"][0]


@pytest.mark.asyncio
async def test_extract_special_conditions():
    payload = _full_payload(special_conditions=[
        "This agreement supersedes all prior NDAs between the parties.",
        "Obligations survive termination for 5 years.",
    ])
    extractor = _make_extractor(_make_llm(payload))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert len(r["special_conditions"]) == 2
    assert "supersedes" in r["special_conditions"][0]


@pytest.mark.asyncio
async def test_extract_contract_value_and_currency():
    payload = _full_payload(contract_value=250000.0, currency="USD")
    extractor = _make_extractor(_make_llm(payload))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert r["contract_value"] == 250000.0
    assert r["currency"] == "USD"


@pytest.mark.asyncio
async def test_extract_liability_cap():
    payload = _full_payload(liability_cap=500000.0, currency="GBP")
    extractor = _make_extractor(_make_llm(payload))
    r = (await extractor.extract(Document(doc_id="doc-001", text="contract text")))[0]

    assert r["liability_cap"] == 500000.0


# ---------------------------------------------------------------------------
# extract() — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_empty_text_returns_none():
    extractor = _make_extractor(_make_llm("{}"))
    result = await extractor.extract(Document(doc_id="doc-002", text="   "))
    assert result is None


@pytest.mark.asyncio
async def test_extract_invalid_json_returns_none():
    extractor = _make_extractor(_make_llm("not json"))
    result = await extractor.extract(Document(doc_id="doc-003", text="contract text"))
    assert result is None


@pytest.mark.asyncio
async def test_extract_json_array_instead_of_object_returns_none():
    """LLM accidentally returns a multi-item non-model array — should return None."""
    extractor = _make_extractor(_make_llm("[1, 2, 3]"))
    result = await extractor.extract(Document(doc_id="doc-004", text="contract text"))
    assert result is None


# ---------------------------------------------------------------------------
# extract() — single-item list unwrap (RecordMode.ONE fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_bare_single_item_list_succeeds():
    """LLM returns [{...}] instead of {...} — unwrapped and accepted."""
    payload = json.dumps([json.loads(_full_payload())])
    extractor = _make_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-SL1", text="contract text"))

    assert result is not None
    assert len(result) == 1
    assert result[0]["contract_type"] == "NDA"
    assert result[0]["effective_date"] == "2024-03-01"


@pytest.mark.asyncio
async def test_extract_bare_single_item_list_sets_doc_id():
    payload = json.dumps([json.loads(_full_payload())])
    extractor = _make_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-SL2", text="contract text"))

    assert result[0]["doc_id"] == "doc-SL2"


@pytest.mark.asyncio
async def test_extract_bare_multi_item_list_returns_none():
    """LLM returns [{...}, {...}] — more than one item, unwrap refuses, returns None."""
    payload = json.dumps([json.loads(_full_payload()), json.loads(_full_payload(contract_type="MSA"))])
    extractor = _make_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-SL3", text="contract text"))

    assert result is None


@pytest.mark.asyncio
async def test_extract_bare_empty_list_returns_none():
    """LLM returns [] — no items, unwrap refuses, returns None."""
    extractor = _make_extractor(_make_llm("[]"))
    result = await extractor.extract(Document(doc_id="doc-SL4", text="contract text"))

    assert result is None


@pytest.mark.asyncio
async def test_extract_missing_fields_default_to_none_or_empty():
    """Minimal response — only the outer object, no fields."""
    extractor = _make_extractor(_make_llm("{}"))
    result = await extractor.extract(Document(doc_id="doc-005", text="contract text"))

    assert result is not None
    r = result[0]
    assert r.get("contract_type") is None
    assert r.get("parties") is None  # absent, not empty list
    assert r.get("key_terms") is None
    assert r.get("special_conditions") is None


@pytest.mark.asyncio
async def test_extract_invalid_numeric_fields_rejects_record():
    """Non-numeric strings for int/float fields fail jsonschema validation — record is None."""
    payload = _full_payload(notice_period_days="thirty", contract_value="one million")
    extractor = _make_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-006", text="contract text"))

    assert result is None


@pytest.mark.asyncio
async def test_extract_explicit_null_fields():
    payload = _full_payload(parties=[], effective_date=None)
    extractor = _make_extractor(_make_llm(payload))
    r = (await extractor.extract(Document(doc_id="doc-007", text="contract text")))[0]

    assert r["parties"] == []
    assert r["effective_date"] is None


@pytest.mark.asyncio
async def test_extract_non_list_key_terms_rejects_record():
    """A scalar where a list is expected fails jsonschema validation — record is None."""
    payload = _full_payload(key_terms="not a list")
    extractor = _make_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-008", text="contract text"))

    assert result is None


@pytest.mark.asyncio
async def test_extract_non_list_special_conditions_rejects_record():
    """A dict where a list is expected fails jsonschema validation — record is None."""
    payload = _full_payload(special_conditions={"key": "value"})
    extractor = _make_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-009", text="contract text"))

    assert result is None


# ---------------------------------------------------------------------------
# extract() — retry behaviour
# ---------------------------------------------------------------------------

def _make_llm_with_responses(*contents: str) -> MagicMock:
    """Build a mock LLMBase that returns each content string in sequence."""
    llm = MagicMock(spec=LLMBase)
    llm.complete = AsyncMock(side_effect=[{"content": c} for c in contents])
    llm.context_window = MagicMock(return_value=128_000)

    async def _stream(*args, **kwargs):
        yield contents[-1]

    llm.complete_stream = _stream
    return llm


@pytest.mark.asyncio
async def test_extract_succeeds_on_retry_after_bad_json(monkeypatch):
    """First call returns bad JSON; second call returns valid JSON — should succeed."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    llm = _make_llm_with_responses("not json", _full_payload())
    extraction_schema = ContractExtraction.model_json_schema()
    extractor = LLMExtractor(
        llm,
        extraction_schema=extraction_schema,
        config=ExtractorConfig(extraction_schema=_DEFAULT_EXTRACTION_SCHEMA, prompt="Extract."),
        record_schema=_build_record_schema(extraction_schema),
        max_retries=2,
    )
    result = await extractor.extract(Document(doc_id="doc-retry-1", text="contract text"))

    assert result is not None and len(result) == 1
    assert llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_extract_returns_none_after_all_retries_exhausted(monkeypatch):
    """All attempts return bad JSON — should return None."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    llm = _make_llm_with_responses("bad", "bad", "bad")
    extraction_schema = ContractExtraction.model_json_schema()
    extractor = LLMExtractor(
        llm,
        extraction_schema=extraction_schema,
        config=ExtractorConfig(extraction_schema=_DEFAULT_EXTRACTION_SCHEMA, prompt="Extract."),
        record_schema=_build_record_schema(extraction_schema),
        max_retries=2,
    )
    result = await extractor.extract(Document(doc_id="doc-retry-2", text="contract text"))

    assert result is None
    assert llm.complete.call_count == 3


@pytest.mark.asyncio
async def test_extract_no_retry_on_success(monkeypatch):
    """First call returns valid JSON — should not retry."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    extraction_schema = ContractExtraction.model_json_schema()
    extractor = LLMExtractor(
        _make_llm(_full_payload()),
        extraction_schema=extraction_schema,
        config=ExtractorConfig(extraction_schema=_DEFAULT_EXTRACTION_SCHEMA, prompt="Extract."),
        record_schema=_build_record_schema(extraction_schema),
        max_retries=2,
    )
    result = await extractor.extract(Document(doc_id="doc-retry-3", text="contract text"))

    assert result is not None and len(result) == 1
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_extract_retry_uses_exponential_backoff(monkeypatch):
    """Sleep is called with 0.2s then 0.4s for two retries."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    llm = _make_llm_with_responses("bad", "bad", _full_payload())
    extraction_schema = ContractExtraction.model_json_schema()
    extractor = LLMExtractor(
        llm,
        extraction_schema=extraction_schema,
        config=ExtractorConfig(extraction_schema=_DEFAULT_EXTRACTION_SCHEMA, prompt="Extract."),
        record_schema=_build_record_schema(extraction_schema),
        max_retries=2,
    )
    await extractor.extract(Document(doc_id="doc-retry-4", text="contract text"))

    assert sleep_mock.call_count == 2
    assert sleep_mock.call_args_list[0].args[0] == pytest.approx(0.2)   # 0.2 * 2^0
    assert sleep_mock.call_args_list[1].args[0] == pytest.approx(0.4)   # 0.2 * 2^1


@pytest.mark.asyncio
async def test_extract_max_retries_zero_no_sleep(monkeypatch):
    """max_retries=0 means one attempt only; no sleep on failure."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    extraction_schema = ContractExtraction.model_json_schema()
    extractor = LLMExtractor(
        _make_llm("bad json"),
        extraction_schema=extraction_schema,
        config=ExtractorConfig(extraction_schema=_DEFAULT_EXTRACTION_SCHEMA, prompt="Extract."),
        record_schema=_build_record_schema(extraction_schema),
        max_retries=0,
    )
    result = await extractor.extract(Document(doc_id="doc-retry-5", text="contract text"))

    assert result is None
    sleep_mock.assert_not_called()


# ---------------------------------------------------------------------------
# extract_as_list=True — helpers
# ---------------------------------------------------------------------------

class _Clause(BaseModel):
    clause_type: str | None = None
    text: str = ""


def _make_list_extractor(
    llm: MagicMock,
    response_field: str = "clauses",
    item_id_field: str = "item_id",
) -> LLMExtractor:
    extraction_schema = _Clause.model_json_schema()
    return LLMExtractor(
        llm,
        extraction_schema=extraction_schema,
        config=ExtractorConfig(
            extraction_schema=_DEFAULT_EXTRACTION_SCHEMA,
            prompt="Extract.",
            record_mode="many",
            response_field=response_field,
            id_field=item_id_field,
            id_template="{doc_id}__{index:04d}",
        ),
        record_schema=_build_list_record_schema(extraction_schema, item_id_field),
    )


def _list_payload(*clauses: dict, field: str = "clauses") -> str:
    return json.dumps({field: list(clauses)})


@pytest.mark.asyncio
async def test_config_driven_list_extractor_builds_prompt_and_injected_fields():
    cfg = ExtractorConfig(
        extraction_schema='{"type":"object","properties":{"text":{"type":"string"}}}',
        record_mode="many",
        response_field="clauses",
        id_field="clause_id",
        id_template="{doc_id}__{index:04d}",
        prompt="Extract all clauses.\n\n",
    )
    extraction_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    record_schema = _build_list_record_schema(extraction_schema, "clause_id")
    extractor = LLMExtractor(
        _make_llm(_list_payload({"text": "Clause text"})),
        extraction_schema=extraction_schema,
        config=cfg,
        record_schema=record_schema,
    )

    assert extractor._record_mode == "many"
    assert extractor._response_field == "clauses"
    assert list(extractor._injected_fields.keys()) == ["doc_id", "clause_id"]
    assert "Extract all clauses." in extractor._system_prompt
    assert '"clauses"' in extractor._system_prompt


def test_config_driven_extractor_rejects_doc_id_in_extraction_schema():
    cfg = ExtractorConfig(
        extraction_schema='{"type":"object","properties":{"doc_id":{"type":"string"}}}',
        prompt="Extract.",
    )
    extraction_schema = {"type": "object", "properties": {"doc_id": {"type": "string"}}}
    record_schema = {"type": "object", "properties": {"doc_id": {"type": "string"}}}

    with pytest.raises(ValueError, match="doc_id"):
        LLMExtractor(
            _make_llm("{}"),
            extraction_schema=extraction_schema,
            config=cfg,
            record_schema=record_schema,
        )


def test_config_driven_extractor_rejects_missing_doc_id_in_record_schema():
    cfg = ExtractorConfig(
        extraction_schema='{"type":"object","properties":{"text":{"type":"string"}}}',
        prompt="Extract.",
        record_mode="many",
        response_field="clauses",
        id_field="clause_id",
        id_template="{doc_id}__{index:04d}",
    )
    extraction_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    record_schema = {"type": "object", "properties": {"clause_id": {"type": "string"}}}

    with pytest.raises(ValueError, match="record schema must include 'doc_id'"):
        LLMExtractor(
            _make_llm("{}"),
            extraction_schema=extraction_schema,
            config=cfg,
            record_schema=record_schema,
        )


# ---------------------------------------------------------------------------
# extract_as_list — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_extract_returns_multiple_records():
    payload = _list_payload(
        {"clause_type": "liability", "text": "Neither party is liable for indirect damages."},
        {"clause_type": "termination", "text": "Either party may terminate with 30 days notice."},
    )
    extractor = _make_list_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-L1", text="contract text"))

    assert result is not None
    assert len(result) == 2


@pytest.mark.asyncio
async def test_list_extract_item_fields():
    payload = _list_payload(
        {"clause_type": "liability", "text": "Neither party is liable for indirect damages."},
        {"clause_type": "termination", "text": "Either party may terminate with 30 days notice."},
    )
    extractor = _make_list_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-L1", text="contract text"))

    assert result[0]["clause_type"] == "liability"
    assert result[0]["text"] == "Neither party is liable for indirect damages."
    assert result[1]["clause_type"] == "termination"


@pytest.mark.asyncio
async def test_list_extract_doc_id_set_on_all_items():
    payload = _list_payload(
        {"clause_type": "payment", "text": "Payment due within 30 days."},
        {"clause_type": "privacy", "text": "Data shall not be shared with third parties."},
    )
    extractor = _make_list_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-L2", text="contract text"))

    assert all(r["doc_id"] == "doc-L2" for r in result)


@pytest.mark.asyncio
async def test_list_extract_item_id_sequential():
    payload = _list_payload(
        {"clause_type": "a", "text": "first"},
        {"clause_type": "b", "text": "second"},
        {"clause_type": "c", "text": "third"},
    )
    extractor = _make_list_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-L3", text="contract text"))

    assert result[0]["item_id"] == "doc-L3__0000"
    assert result[1]["item_id"] == "doc-L3__0001"
    assert result[2]["item_id"] == "doc-L3__0002"


@pytest.mark.asyncio
async def test_list_extract_single_item():
    payload = _list_payload({"clause_type": "governing_law", "text": "Laws of England and Wales."})
    extractor = _make_list_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-L4", text="contract text"))

    assert len(result) == 1
    assert result[0]["clause_type"] == "governing_law"


@pytest.mark.asyncio
async def test_list_extract_empty_array_returns_empty_list():
    """LLM returns an empty array — valid result, not a parse failure."""
    payload = _list_payload(field="clauses")  # no items
    extractor = _make_list_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-L5", text="contract text"))

    assert result == []


@pytest.mark.asyncio
async def test_list_extract_custom_response_field():
    payload = json.dumps({"items": [{"clause_type": "ip", "text": "All IP is retained by licensor."}]})
    extractor = _make_list_extractor(_make_llm(payload), response_field="items")
    result = await extractor.extract(Document(doc_id="doc-L6", text="contract text"))

    assert len(result) == 1
    assert result[0]["clause_type"] == "ip"


# ---------------------------------------------------------------------------
# extract_as_list — error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_extract_invalid_json_returns_none():
    extractor = _make_list_extractor(_make_llm("not json"))
    result = await extractor.extract(Document(doc_id="doc-LE1", text="contract text"))
    assert result is None


@pytest.mark.asyncio
async def test_list_extract_wrong_wrapper_key_returns_none():
    """JSON has a different key than expected — missing response_field."""
    payload = json.dumps({"wrong_key": [{"clause_type": "payment", "text": "..."}]})
    extractor = _make_list_extractor(_make_llm(payload))
    result = await extractor.extract(Document(doc_id="doc-LE2", text="contract text"))
    assert result is None


@pytest.mark.asyncio
async def test_list_extract_blank_text_returns_none():
    extractor = _make_list_extractor(_make_llm(_list_payload()))
    result = await extractor.extract(Document(doc_id="doc-LE3", text="   "))
    assert result is None


@pytest.mark.asyncio
async def test_list_extract_retry_on_bad_json(monkeypatch):
    """First call returns bad JSON; second succeeds — retry works in list mode."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    payload = _list_payload({"clause_type": "termination", "text": "30 days notice."})
    llm = _make_llm_with_responses("bad json", payload)
    extraction_schema = _Clause.model_json_schema()
    extractor = LLMExtractor(
        llm,
        extraction_schema=extraction_schema,
        config=ExtractorConfig(
            extraction_schema=_DEFAULT_EXTRACTION_SCHEMA,
            prompt="Extract.",
            record_mode="many",
            response_field="clauses",
            id_field="item_id",
            id_template="{doc_id}__{index:04d}",
        ),
        record_schema=_build_list_record_schema(extraction_schema, "item_id"),
        max_retries=1,
    )
    result = await extractor.extract(Document(doc_id="doc-LR1", text="contract text"))

    assert result is not None and len(result) == 1
    assert llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# item_id_field customisation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_item_id_field_on_extracted_records():
    payload = _list_payload(
        {"clause_type": "liability", "text": "Clause A."},
        {"clause_type": "termination", "text": "Clause B."},
    )
    extractor = _make_list_extractor(_make_llm(payload), item_id_field="clause_id")
    result = await extractor.extract(Document(doc_id="doc-CID1", text="contract text"))

    assert result is not None and len(result) == 2
    assert result[0]["clause_id"] == "doc-CID1__0000"
    assert result[1]["clause_id"] == "doc-CID1__0001"


@pytest.mark.asyncio
async def test_custom_item_id_field_records_have_no_item_id_attr():
    """Records built with clause_id should not have an item_id key."""
    payload = _list_payload({"clause_type": "ip", "text": "All IP retained by licensor."})
    extractor = _make_list_extractor(_make_llm(payload), item_id_field="clause_id")
    result = await extractor.extract(Document(doc_id="doc-CID2", text="contract text"))

    assert result is not None
    assert "item_id" not in result[0]


# ---------------------------------------------------------------------------
# Mini-model usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_calls_llm_with_mini_model() -> None:
    llm = _make_llm(_full_payload())
    extractor = _make_extractor(llm)

    await extractor.extract(Document(doc_id="doc-mini", text="contract text"))

    llm.complete.assert_called_once()
    assert llm.complete.call_args.kwargs.get("model") == "mini"


# ---------------------------------------------------------------------------
# Windowing — long documents split into multiple LLM calls and merge
# ---------------------------------------------------------------------------


def _force_windows(monkeypatch, n: int) -> None:
    """Patch the splitter so any document is split into exactly *n* windows."""
    monkeypatch.setattr(
        "cogbase.pipeline.extraction.llm.split_text_by_tokens",
        lambda text, budget, overlap: [text] * n,
    )


@pytest.mark.asyncio
async def test_window_budget_sizes_from_mini_context_window(monkeypatch) -> None:
    """The per-window budget is derived from the mini model's context window."""
    captured: list[int] = []

    def _spy_split(text, budget, overlap):
        captured.append(budget)
        return [text]

    monkeypatch.setattr("cogbase.pipeline.extraction.llm.split_text_by_tokens", _spy_split)
    llm = _make_llm(_full_payload())
    llm.context_window = MagicMock(return_value=8_000)
    extractor = _make_extractor(llm)

    await extractor.extract(Document(doc_id="doc-w0", text="contract text"))

    llm.context_window.assert_called_with("mini")
    # 0.5 * 8000 minus the system-prompt tokens — well above the 1000 floor.
    assert 1_000 <= captured[0] <= 4_000


@pytest.mark.asyncio
async def test_short_document_makes_one_llm_call() -> None:
    """A document that fits one window triggers exactly one extraction call."""
    llm = _make_llm(_full_payload())
    extractor = _make_extractor(llm)

    await extractor.extract(Document(doc_id="doc-w1", text="contract text"))

    assert llm.complete.call_count == 1


@pytest.mark.asyncio
async def test_many_mode_concatenates_records_across_windows(monkeypatch) -> None:
    """List mode maps over windows and concatenates the per-window record lists."""
    _force_windows(monkeypatch, 2)
    llm = _make_llm_with_responses(
        _list_payload({"clause_type": "a", "text": "first"}, {"clause_type": "b", "text": "second"}),
        _list_payload({"clause_type": "c", "text": "third"}),
    )
    extractor = _make_list_extractor(llm)
    result = await extractor.extract(Document(doc_id="doc-w2", text="long contract"))

    assert llm.complete.call_count == 2
    assert [r["clause_type"] for r in result] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_many_mode_reindexes_ids_across_merged_windows(monkeypatch) -> None:
    """Item ids index across the merged list, not restarting per window."""
    _force_windows(monkeypatch, 2)
    llm = _make_llm_with_responses(
        _list_payload({"clause_type": "a", "text": "first"}, {"clause_type": "b", "text": "second"}),
        _list_payload({"clause_type": "c", "text": "third"}),
    )
    extractor = _make_list_extractor(llm)
    result = await extractor.extract(Document(doc_id="doc-w3", text="long contract"))

    assert [r["item_id"] for r in result] == [
        "doc-w3__0000",
        "doc-w3__0001",
        "doc-w3__0002",
    ]
    assert all(r["doc_id"] == "doc-w3" for r in result)


@pytest.mark.asyncio
async def test_many_mode_window_parse_failure_fails_extraction(monkeypatch) -> None:
    """A single bad window fails the whole attempt."""
    _force_windows(monkeypatch, 2)
    llm = _make_llm_with_responses(
        _list_payload({"clause_type": "a", "text": "first"}),
        "not json",
    )
    extraction_schema = _Clause.model_json_schema()
    extractor = LLMExtractor(
        llm,
        extraction_schema=extraction_schema,
        config=ExtractorConfig(
            extraction_schema=_DEFAULT_EXTRACTION_SCHEMA,
            prompt="Extract.",
            record_mode="many",
            response_field="clauses",
            id_field="item_id",
            id_template="{doc_id}__{index:04d}",
        ),
        record_schema=_build_list_record_schema(extraction_schema, "item_id"),
        max_retries=0,
    )
    result = await extractor.extract(Document(doc_id="doc-w4", text="long contract"))

    assert result is None


@pytest.mark.asyncio
async def test_one_mode_reconciles_scalar_fields_across_windows(monkeypatch) -> None:
    """Single mode reduces windows: each field takes the first non-null value."""
    _force_windows(monkeypatch, 2)
    llm = _make_llm_with_responses(
        _full_payload(contract_type="NDA", governing_law=None),
        _full_payload(contract_type=None, governing_law="Laws of New York."),
    )
    extractor = _make_extractor(llm)
    r = (await extractor.extract(Document(doc_id="doc-w5", text="long contract")))[0]

    assert llm.complete.call_count == 2
    assert r["contract_type"] == "NDA"            # from window 1
    assert r["governing_law"] == "Laws of New York."  # from window 2
    assert r["doc_id"] == "doc-w5"


@pytest.mark.asyncio
async def test_one_mode_concatenates_list_fields_across_windows(monkeypatch) -> None:
    """List-valued fields on a single record are concatenated across windows."""
    _force_windows(monkeypatch, 2)
    llm = _make_llm_with_responses(
        _full_payload(parties=[{"name": "Acme Corp", "role": "discloser", "jurisdiction": None}]),
        _full_payload(parties=[{"name": "Supplier Ltd", "role": "recipient", "jurisdiction": None}]),
    )
    extractor = _make_extractor(llm)
    r = (await extractor.extract(Document(doc_id="doc-w6", text="long contract")))[0]

    assert [p["name"] for p in r["parties"]] == ["Acme Corp", "Supplier Ltd"]


@pytest.mark.asyncio
async def test_one_mode_single_window_passes_through_unchanged(monkeypatch) -> None:
    """A single window in single mode reconciles to itself with no extra call."""
    _force_windows(monkeypatch, 1)
    llm = _make_llm(_full_payload(contract_type="MSA"))
    extractor = _make_extractor(llm)
    r = (await extractor.extract(Document(doc_id="doc-w7", text="contract text")))[0]

    assert llm.complete.call_count == 1
    assert r["contract_type"] == "MSA"


# ---------------------------------------------------------------------------
# Overlap-induced duplicate handling in the merge step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_many_mode_dedups_overlap_duplicate_across_adjacent_windows(monkeypatch) -> None:
    """A clause re-extracted from the overlap of two adjacent windows is dropped once."""
    _force_windows(monkeypatch, 2)
    llm = _make_llm_with_responses(
        _list_payload({"clause_type": "a", "text": "first"}, {"clause_type": "b", "text": "second"}),
        _list_payload({"clause_type": "b", "text": "second"}, {"clause_type": "c", "text": "third"}),
    )
    extractor = _make_list_extractor(llm)
    result = await extractor.extract(Document(doc_id="doc-d1", text="long contract"))

    assert [r["clause_type"] for r in result] == ["a", "b", "c"]
    assert [r["item_id"] for r in result] == ["doc-d1__0000", "doc-d1__0001", "doc-d1__0002"]


@pytest.mark.asyncio
async def test_many_mode_dedup_ignores_whitespace_differences(monkeypatch) -> None:
    """Overlap duplicates that differ only in whitespace are still de-duplicated."""
    _force_windows(monkeypatch, 2)
    llm = _make_llm_with_responses(
        _list_payload({"clause_type": "x", "text": "Clause one."}),
        _list_payload({"clause_type": "x", "text": "Clause   one."}),
    )
    extractor = _make_list_extractor(llm)
    result = await extractor.extract(Document(doc_id="doc-d2", text="long contract"))

    assert len(result) == 1


@pytest.mark.asyncio
async def test_many_mode_keeps_intra_window_duplicates(monkeypatch) -> None:
    """Two identical records the model returns within one window are both kept."""
    _force_windows(monkeypatch, 1)
    llm = _make_llm(
        _list_payload({"clause_type": "a", "text": "same"}, {"clause_type": "a", "text": "same"})
    )
    extractor = _make_list_extractor(llm)
    result = await extractor.extract(Document(doc_id="doc-d3", text="contract"))

    assert len(result) == 2


@pytest.mark.asyncio
async def test_many_mode_keeps_duplicate_in_non_adjacent_windows(monkeypatch) -> None:
    """An identical record recurring in non-adjacent windows is genuine, not overlap."""
    _force_windows(monkeypatch, 3)
    llm = _make_llm_with_responses(
        _list_payload({"clause_type": "a", "text": "repeat"}),
        _list_payload({"clause_type": "b", "text": "middle"}),
        _list_payload({"clause_type": "a", "text": "repeat"}),
    )
    extractor = _make_list_extractor(llm)
    result = await extractor.extract(Document(doc_id="doc-d4", text="long contract"))

    assert [r["clause_type"] for r in result] == ["a", "b", "a"]


@pytest.mark.asyncio
async def test_one_mode_dedups_list_field_across_overlap(monkeypatch) -> None:
    """A list field's overlap-duplicated element is gathered once when reconciled."""
    _force_windows(monkeypatch, 2)
    a = {"name": "Acme Corp", "role": "discloser", "jurisdiction": None}
    b = {"name": "Supplier Ltd", "role": "recipient", "jurisdiction": None}
    c = {"name": "Holdco", "role": "guarantor", "jurisdiction": None}
    llm = _make_llm_with_responses(
        _full_payload(parties=[a, b]),
        _full_payload(parties=[b, c]),
    )
    extractor = _make_extractor(llm)
    r = (await extractor.extract(Document(doc_id="doc-d5", text="long contract")))[0]

    assert [p["name"] for p in r["parties"]] == ["Acme Corp", "Supplier Ltd", "Holdco"]


# ---------------------------------------------------------------------------
# split_text_by_tokens — token-bounded, overlapping recursive splitter
# ---------------------------------------------------------------------------


def test_split_text_by_tokens_short_text_is_single_window() -> None:
    assert split_text_by_tokens("a short clause", 1_000, 100) == ["a short clause"]


def test_split_text_by_tokens_windows_are_token_bounded_and_cover_paragraphs() -> None:
    paras = [f"Paragraph {i}: " + " ".join(f"word{i}x{j}" for j in range(20)) for i in range(6)]
    text = "\n\n".join(paras)
    budget = estimate_tokens(paras[0]) * 2 + 4  # room for ~2 paragraphs per window
    overlap = budget // 5

    windows = split_text_by_tokens(text, budget, overlap)

    assert len(windows) >= 2
    # Sized in tokens, not characters: no window exceeds the token budget.
    assert all(estimate_tokens(w) <= budget for w in windows)
    # Every paragraph appears intact in some window (cuts land between paragraphs).
    assert all(any(p in w for w in windows) for p in paras)


def test_split_text_by_tokens_overlaps_consecutive_windows() -> None:
    paras = [f"Clause {i}: unique alpha{i} beta{i} gamma{i} delta{i} epsilon{i}." for i in range(8)]
    text = "\n\n".join(paras)
    budget = estimate_tokens(paras[0]) * 2 + 2
    overlap = estimate_tokens(paras[0])  # ~one paragraph of overlap

    windows = split_text_by_tokens(text, budget, overlap)

    assert len(windows) >= 2
    # With paragraph-sized overlap, at least one paragraph is shared by two windows.
    shared = [p for p in paras if sum(p in w for w in windows) >= 2]
    assert shared
