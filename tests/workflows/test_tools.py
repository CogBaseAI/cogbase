"""Unit tests for cogbase.workflows.tools — the four built-in step tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from pydantic import TypeAdapter, ValidationError

from cogbase.config.config import WorkflowStepConfig
from cogbase.core.models import Chunk
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from cogbase.stores import CollectionSchema, VectorCollectionSchema
from cogbase.stores.schema import FieldSchema, FieldType

from cogbase.workflows.tools.structured_query import run as sq_run
from cogbase.workflows.tools.vector_search import run as vs_run
from cogbase.workflows.tools.llm_structured import run as ls_run
from cogbase.workflows.tools.structured_save import run as ss_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLAUSE_SCHEMA = CollectionSchema(
    name="clauses",
    primary_fields=["clause_id"],
    description="Test clause collection",
    fields={
        "clause_id": FieldSchema(type=FieldType.STRING),
        "text":      FieldSchema(type=FieldType.STRING),
    },
)

_FINDING_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "finding_id": {"type": "string"},
        "status":     {"type": "string"},
    },
    "required": ["finding_id", "status"],
})


class _Finding(BaseModel):
    finding_id: str
    status: str


async def _make_structured_store(*records: dict) -> InMemoryStructuredStore:
    store = InMemoryStructuredStore()
    await store.create_collection(_CLAUSE_SCHEMA)
    if records:
        from pydantic import create_model
        Row = create_model("Row", clause_id=(str, ...), text=(str, ""))
        await store.save("clauses", [Row(**r) for r in records])
    return store


async def _make_vector_store(dim: int = 4) -> FAISSVectorStore:
    vs = FAISSVectorStore()
    schema = VectorCollectionSchema(name="rules", dimensions=dim, description="rule chunks")
    await vs.create_collection(schema)
    return vs


_STEP_ADAPTER: TypeAdapter[WorkflowStepConfig] = TypeAdapter(WorkflowStepConfig)


def _make_step(**kwargs) -> WorkflowStepConfig:
    return _STEP_ADAPTER.validate_python({"id": "test-step", **kwargs})


def _make_llm(response: str, context_window: int = 128_000) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": response})
    llm.context_window = MagicMock(return_value=context_window)
    return llm


def _make_embedder(dim: int = 4) -> MagicMock:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1] * dim])
    return embedder


# ---------------------------------------------------------------------------
# structured-query
# ---------------------------------------------------------------------------

class TestStructuredQueryTool:
    async def test_returns_all_records_without_filters(self):
        store = await _make_structured_store(
            {"clause_id": "c1", "text": "alpha"},
            {"clause_id": "c2", "text": "beta"},
        )
        step = _make_step(tool="structured-query", collection="clauses")
        output = await sq_run(step, {}, store)
        assert len(output["records"]) == 2

    async def test_eq_filter_applied(self):
        store = await _make_structured_store(
            {"clause_id": "c1", "text": "alpha"},
            {"clause_id": "c2", "text": "beta"},
        )
        step = _make_step(
            tool="structured-query",
            collection="clauses",
            filters={"clause_id": "c1"},
        )
        output = await sq_run(step, {}, store)
        assert len(output["records"]) == 1
        assert output["records"][0]["clause_id"] == "c1"

    async def test_filter_value_is_template_rendered(self):
        store = await _make_structured_store({"clause_id": "c1", "text": "alpha"})
        step = _make_step(
            tool="structured-query",
            collection="clauses",
            filters={"clause_id": "{{ input.doc_id }}"},
        )
        ctx = {"input": {"doc_id": "c1"}, "steps": {}}
        output = await sq_run(step, ctx, store)
        assert len(output["records"]) == 1

    async def test_empty_collection_returns_empty_list(self):
        store = await _make_structured_store()
        step = _make_step(tool="structured-query", collection="clauses")
        output = await sq_run(step, {}, store)
        assert output["records"] == []

    async def test_missing_store_raises(self):
        step = _make_step(tool="structured-query", collection="clauses")
        with pytest.raises(RuntimeError, match="structured store"):
            await sq_run(step, {}, None)

    def test_missing_collection_raises(self):
        with pytest.raises(ValidationError):
            _make_step(tool="structured-query")

    async def test_min_records_raises_when_the_query_comes_back_short(self):
        """An empty reference collection must fail the run, not empty it.

        Without this the foreach downstream iterates zero times, nothing is saved,
        and the workflow *succeeds* — a run that never checked anything is
        indistinguishable from one that found nothing to report.
        """
        store = await _make_structured_store()
        step = _make_step(tool="structured-query", collection="clauses", min_records=1)
        with pytest.raises(RuntimeError, match="min_records=1"):
            await sq_run(step, {}, store)

    async def test_min_records_reports_the_rendered_filter_values(self):
        """The template alone does not say what it rendered to, and a value that
        matches no row is the usual cause."""
        store = await _make_structured_store({"clause_id": "c1", "text": "alpha"})
        step = _make_step(
            tool="structured-query",
            collection="clauses",
            filters={"clause_id": "{{ input.doc_id }}"},
            min_records=1,
        )
        ctx = {"input": {"doc_id": "nonexistent"}, "steps": {}}
        with pytest.raises(RuntimeError, match="nonexistent"):
            await sq_run(step, ctx, store)

    async def test_min_records_is_satisfied_by_exactly_that_many(self):
        store = await _make_structured_store({"clause_id": "c1", "text": "alpha"})
        step = _make_step(tool="structured-query", collection="clauses", min_records=1)
        output = await sq_run(step, {}, store)
        assert len(output["records"]) == 1

    async def test_min_records_defaults_to_allowing_an_empty_result(self):
        """Every config written before the field existed keeps its behaviour."""
        store = await _make_structured_store()
        step = _make_step(tool="structured-query", collection="clauses")
        assert step.min_records == 0
        assert (await sq_run(step, {}, store))["records"] == []

    def test_negative_min_records_is_rejected(self):
        with pytest.raises(ValidationError):
            _make_step(tool="structured-query", collection="clauses", min_records=-1)


# ---------------------------------------------------------------------------
# vector-search
# ---------------------------------------------------------------------------

class TestVectorSearchTool:
    async def test_returns_chunks_key(self):
        vs = await _make_vector_store()
        embedder = _make_embedder()
        step = _make_step(tool="vector-search", collection="rules", query="test query", top_k=3)
        output = await vs_run(step, {}, vs, embedder)
        assert "chunks" in output
        assert isinstance(output["chunks"], list)

    async def test_embedder_called_with_rendered_query(self):
        vs = await _make_vector_store()
        embedder = _make_embedder()
        ctx = {"item": {"clause_type": "liability", "text": "Vendor liable..."}}
        step = _make_step(
            tool="vector-search",
            collection="rules",
            query="{{ item.clause_type }}\n{{ item.text }}",
            top_k=2,
        )
        await vs_run(step, ctx, vs, embedder)
        embedder.embed.assert_called_once()
        call_args = embedder.embed.call_args[0][0]
        assert call_args[0] == "liability\nVendor liable..."

    async def test_top_k_passed_to_store(self):
        vs = MagicMock()
        vs.search = AsyncMock(return_value=[])
        embedder = _make_embedder()
        step = _make_step(tool="vector-search", collection="rules", query="q", top_k=7)
        await vs_run(step, {}, vs, embedder)
        vs.search.assert_called_once()
        _, kwargs = vs.search.call_args
        # top_k passed as positional or keyword
        call_args = vs.search.call_args
        assert 7 in call_args.args or call_args.kwargs.get("top_k") == 7

    async def test_missing_vector_store_raises(self):
        step = _make_step(tool="vector-search", collection="rules", query="q")
        with pytest.raises(RuntimeError, match="vector store"):
            await vs_run(step, {}, None, _make_embedder())

    async def test_missing_embedder_raises(self):
        vs = await _make_vector_store()
        step = _make_step(tool="vector-search", collection="rules", query="q")
        with pytest.raises(RuntimeError, match="embedder"):
            await vs_run(step, {}, vs, None)

    def test_missing_collection_raises(self):
        with pytest.raises(ValidationError):
            _make_step(tool="vector-search", query="q")

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            _make_step(tool="vector-search", collection="rules")

    async def test_returned_chunks_carry_no_embedding(self):
        """The store populates it; the next step almost always JSON-serializes the
        chunks into an LLM input, where 1536 floats per chunk are pure spend."""
        vs = await _make_vector_store()
        await vs.upsert("rules", [
            Chunk(chunk_id="a_0", doc_id="doc-a", text="alpha", embedding=[0.1] * 4),
        ])
        step = _make_step(tool="vector-search", collection="rules", query="q", top_k=3)
        output = await vs_run(step, {}, vs, _make_embedder())
        assert [c.chunk_id for c in output["chunks"]] == ["a_0"]
        assert all(c.embedding is None for c in output["chunks"])

    async def test_stripping_the_embedding_leaves_the_stored_chunk_intact(self):
        """A copy, not a mutation — the FAISS store hands out its own Chunk objects,
        so clearing the field in place would empty the collection's vectors."""
        vs = await _make_vector_store()
        await vs.upsert("rules", [
            Chunk(chunk_id="a_0", doc_id="doc-a", text="alpha", embedding=[0.1] * 4),
        ])
        step = _make_step(tool="vector-search", collection="rules", query="q", top_k=3)
        await vs_run(step, {}, vs, _make_embedder())
        again = await vs.search("rules", "q", [0.1] * 4, top_k=3)
        assert again[0].embedding == [0.1] * 4

    async def test_doc_id_filter_scopes_results(self):
        """A `filters: {doc_id: ...}` step only returns chunks from that document."""
        vs = await _make_vector_store()
        await vs.upsert("rules", [
            Chunk(chunk_id="a_0", doc_id="doc-a", text="alpha", embedding=[0.1] * 4),
            Chunk(chunk_id="b_0", doc_id="doc-b", text="beta", embedding=[0.1] * 4),
        ])
        embedder = _make_embedder()
        ctx = {"input": {"doc_id": "doc-a"}}
        step = _make_step(
            tool="vector-search",
            collection="rules",
            query="q",
            top_k=5,
            filters={"doc_id": "{{ input.doc_id }}"},
        )
        output = await vs_run(step, ctx, vs, embedder)
        assert output["chunks"]
        assert all(c.doc_id == "doc-a" for c in output["chunks"])

    async def test_no_filters_returns_cross_document_results(self):
        """Without `filters:`, results are not scoped to a single document (no accidental default)."""
        vs = await _make_vector_store()
        await vs.upsert("rules", [
            Chunk(chunk_id="a_0", doc_id="doc-a", text="alpha", embedding=[0.1] * 4),
            Chunk(chunk_id="b_0", doc_id="doc-b", text="beta", embedding=[0.1] * 4),
        ])
        embedder = _make_embedder()
        step = _make_step(tool="vector-search", collection="rules", query="q", top_k=5)
        output = await vs_run(step, {}, vs, embedder)
        assert {c.doc_id for c in output["chunks"]} == {"doc-a", "doc-b"}

    async def test_filters_passed_to_store_search(self):
        vs = MagicMock()
        vs.search = AsyncMock(return_value=[])
        embedder = _make_embedder()
        ctx = {"input": {"doc_id": "doc-a"}}
        step = _make_step(
            tool="vector-search",
            collection="rules",
            query="q",
            filters={"doc_id": "{{ input.doc_id }}"},
        )
        await vs_run(step, ctx, vs, embedder)
        _, kwargs = vs.search.call_args
        assert kwargs["filters"] is not None
        assert len(kwargs["filters"]) == 1


# ---------------------------------------------------------------------------
# llm-structured
# ---------------------------------------------------------------------------

class TestLLMStructuredTool:
    async def test_parses_valid_response(self):
        llm = _make_llm('{"finding_id": "f1", "status": "compliant"}')
        step = _make_step(
            tool="llm-structured",
            prompt="You are a judge.",
            input={"data": "some input"},
            output_schema=_FINDING_JSON_SCHEMA,
        )
        output = await ls_run(step, {}, llm)
        assert "output" in output
        result = output["output"]
        assert result["finding_id"] == "f1"
        assert result["status"] == "compliant"

    async def test_input_values_are_template_rendered(self):
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}')
        ctx = {"item": {"clause_id": "c1", "text": "Vendor shall..."}}
        step = _make_step(
            tool="llm-structured",
            prompt="Judge this.",
            input={"clause": "{{ item }}"},
            output_schema=_FINDING_JSON_SCHEMA,
        )
        await ls_run(step, ctx, llm)
        call_args = llm.complete.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "c1" in user_msg
        assert "Vendor shall" in user_msg

    async def test_prompt_used_as_system_message(self):
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}')
        step = _make_step(
            tool="llm-structured",
            prompt="You are a compliance expert.",
            output_schema=_FINDING_JSON_SCHEMA,
        )
        await ls_run(step, {}, llm)
        messages = llm.complete.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("You are a compliance expert.")

    async def test_schema_hint_in_system_message(self):
        """Schema is injected into the system message so the LLM knows the expected shape."""
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}')
        step = _make_step(
            tool="llm-structured",
            prompt="Judge this.",
            output_schema=_FINDING_JSON_SCHEMA,
        )
        await ls_run(step, {}, llm)
        messages = llm.complete.call_args[0][0]
        system_msg = messages[0]["content"]
        assert "finding_id" in system_msg
        assert "status" in system_msg
        assert "JSON Schema" in system_msg

    async def test_llm_called_with_zero_temperature(self):
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}')
        step = _make_step(
            tool="llm-structured",
            prompt="Judge.",
            output_schema=_FINDING_JSON_SCHEMA,
        )
        await ls_run(step, {}, llm)
        _, kwargs = llm.complete.call_args
        assert kwargs.get("temperature") == 0.0

    async def test_missing_llm_raises(self):
        step = _make_step(tool="llm-structured", prompt="x", output_schema=_FINDING_JSON_SCHEMA)
        with pytest.raises(RuntimeError, match="LLM"):
            await ls_run(step, {}, None)

    async def test_input_over_context_budget_raises_without_calling_llm(self):
        """A rendered input over the context-window budget fails fast, before the LLM call."""
        # Tiny window so a modest input overflows the 80% budget.
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}', context_window=100)
        big = "word " * 500  # ~625 est. tokens, well over the 80-token budget
        step = _make_step(
            tool="llm-structured",
            prompt="Judge these.",
            input={"facts": big},
            output_schema=_FINDING_JSON_SCHEMA,
        )
        with pytest.raises(ValueError, match="over the ~80-token budget"):
            await ls_run(step, {}, llm)
        llm.complete.assert_not_called()

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            _make_step(tool="llm-structured", output_schema=_FINDING_JSON_SCHEMA)

    def test_missing_output_schema_raises(self):
        with pytest.raises(ValidationError):
            _make_step(tool="llm-structured", prompt="x")

    async def test_empty_llm_response_raises(self):
        step = _make_step(
            tool="llm-structured",
            prompt="x",
            output_schema=_FINDING_JSON_SCHEMA,
        )
        with pytest.raises(ValueError, match="empty"):
            await ls_run(step, {}, _make_llm(""))

    async def test_invalid_json_response_raises(self):
        step = _make_step(
            tool="llm-structured",
            prompt="x",
            output_schema=_FINDING_JSON_SCHEMA,
        )
        with pytest.raises(ValueError, match="failed to parse"):
            await ls_run(step, {}, _make_llm("not valid json"))

    async def test_schema_violation_raises_after_retries(self):
        """A well-formed JSON that fails schema validation exhausts retries."""
        step = _make_step(
            tool="llm-structured",
            prompt="x",
            output_schema=_FINDING_JSON_SCHEMA,
        )
        # Missing required "status" field
        with pytest.raises(ValueError, match="failed to parse"):
            await ls_run(step, {}, _make_llm('{"finding_id": "f1"}'))

    # --- null-type schema variations ---

    _NULLABLE_SCHEMA = json.dumps({
        "type": "object",
        "properties": {
            "id":    {"type": "string"},
            "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "required": ["id"],
    })

    async def test_nullable_field_accepts_null(self):
        """anyOf [string, null] must accept JSON null without validation error."""
        step = _make_step(
            tool="llm-structured",
            prompt="x",
            output_schema=self._NULLABLE_SCHEMA,
        )
        output = await ls_run(step, {}, _make_llm('{"id": "r1", "title": null}'))
        assert output["output"]["title"] is None

    async def test_nullable_field_accepts_string(self):
        step = _make_step(
            tool="llm-structured",
            prompt="x",
            output_schema=self._NULLABLE_SCHEMA,
        )
        output = await ls_run(step, {}, _make_llm('{"id": "r1", "title": "Introduction"}'))
        assert output["output"]["title"] == "Introduction"

    async def test_nullable_number_field_accepts_null(self):
        step = _make_step(
            tool="llm-structured",
            prompt="x",
            output_schema=self._NULLABLE_SCHEMA,
        )
        output = await ls_run(step, {}, _make_llm('{"id": "r1", "score": null}'))
        assert output["output"]["score"] is None



# ---------------------------------------------------------------------------
# structured-save
# ---------------------------------------------------------------------------

class TestStructuredSaveTool:
    async def _make_finding_store(self) -> InMemoryStructuredStore:
        schema = CollectionSchema(
            name="findings",
            primary_fields=["finding_id"],
            description="Test findings collection",
            fields={
                "finding_id": FieldSchema(type=FieldType.STRING),
                "status":     FieldSchema(type=FieldType.STRING),
            },
        )
        store = InMemoryStructuredStore()
        await store.create_collection(schema)
        return store

    async def test_saves_pydantic_model_record(self):
        store = await self._make_finding_store()
        finding = _Finding(finding_id="f1", status="compliant")
        ctx = {"steps": {"judge": {"output": finding}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records=["{{ steps.judge.output }}"],
        )
        output = await ss_run(step, ctx, store)
        assert len(output["records"]) == 1
        rows = await store.query("findings")
        assert len(rows) == 1
        assert rows[0]["finding_id"] == "f1"

    async def test_returns_records_in_output(self):
        store = await self._make_finding_store()
        finding = _Finding(finding_id="f2", status="non_compliant")
        ctx = {"steps": {"judge": {"output": finding}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records=["{{ steps.judge.output }}"],
        )
        output = await ss_run(step, ctx, store)
        assert output["records"][0] is finding

    async def test_saves_dict_record(self):
        """structured-save must accept plain dicts (e.g. llm-structured output)."""
        store = await self._make_finding_store()
        ctx = {"steps": {"judge": {"output": {"finding_id": "f3", "status": "compliant"}}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records=["{{ steps.judge.output }}"],
        )
        await ss_run(step, ctx, store)
        rows = await store.query("findings")
        assert len(rows) == 1
        assert rows[0]["finding_id"] == "f3"

    async def test_empty_records_from_skips_save(self):
        """records_from resolving to an empty list (e.g. judge found nothing) skips save."""
        store = MagicMock(spec=InMemoryStructuredStore)
        store.save = AsyncMock()
        ctx = {"steps": {"judge": {"output": {"findings": []}}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records_from="{{ steps.judge.output.findings }}",
        )
        output = await ss_run(step, ctx, store)
        store.save.assert_not_called()
        assert output["records"] == []

    async def test_records_from_saves_list(self):
        """records_from resolving to a list saves every element as its own record."""
        store = await self._make_finding_store()
        ctx = {"steps": {"judge": {"output": {"findings": [
            {"finding_id": "f1", "status": "compliant"},
            {"finding_id": "f2", "status": "non_compliant"},
        ]}}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records_from="{{ steps.judge.output.findings }}",
        )
        output = await ss_run(step, ctx, store)
        assert len(output["records"]) == 2
        rows = await store.query("findings")
        assert {r["finding_id"] for r in rows} == {"f1", "f2"}

    async def test_records_from_non_list_raises(self):
        store = await self._make_finding_store()
        ctx = {"steps": {"judge": {"output": {"finding_id": "f1", "status": "compliant"}}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records_from="{{ steps.judge.output }}",
        )
        with pytest.raises(RuntimeError, match="must resolve to a list"):
            await ss_run(step, ctx, store)

    async def test_missing_store_raises(self):
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records=["{{ steps.judge.output }}"],
        )
        with pytest.raises(RuntimeError, match="structured store"):
            await ss_run(step, {}, None)

    def test_missing_collection_raises(self):
        with pytest.raises(ValidationError):
            _make_step(tool="structured-save")

    async def test_fields_overlay_overrides_model_output(self):
        """`fields:` values win over same-named keys the LLM emitted — the citation-precision fix."""
        store = await self._make_finding_store()
        finding = _Finding(finding_id="wrong", status="compliant")
        ctx = {"input": {"doc_id": "doc-a"}, "steps": {"judge": {"output": finding}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records=["{{ steps.judge.output }}"],
            fields={"finding_id": "{{ input.doc_id }}"},
        )
        await ss_run(step, ctx, store)
        rows = await store.query("findings")
        assert len(rows) == 1
        assert rows[0]["finding_id"] == "doc-a"

    async def test_no_fields_is_a_noop(self):
        """A save step with no `fields:` declared round-trips the record unchanged."""
        store = await self._make_finding_store()
        finding = _Finding(finding_id="f1", status="compliant")
        ctx = {"steps": {"judge": {"output": finding}}}
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records=["{{ steps.judge.output }}"],
        )
        output = await ss_run(step, ctx, store)
        assert output["records"][0] is finding

    async def test_fields_adds_keys_not_on_the_record(self):
        """`fields:` keys absent from the record are added, not dropped."""
        store = await self._make_finding_store()
        ctx = {
            "input": {"doc_id": "doc-a"},
            "steps": {"judge": {"output": {"finding_id": "f1", "status": "compliant"}}},
        }
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records=["{{ steps.judge.output }}"],
            fields={"doc_id": "{{ input.doc_id }}"},
        )
        output = await ss_run(step, ctx, store)
        assert output["records"][0]["doc_id"] == "doc-a"
        assert output["records"][0]["finding_id"] == "f1"

    async def test_fields_overlay_applied_to_records_from(self):
        store = await self._make_finding_store()
        ctx = {
            "input": {"doc_id": "doc-a"},
            "steps": {"judge": {"output": {"findings": [
                {"finding_id": "f1", "status": "compliant"},
                {"finding_id": "f2", "status": "non_compliant"},
            ]}}},
        }
        step = _make_step(
            tool="structured-save",
            collection="findings",
            records_from="{{ steps.judge.output.findings }}",
            fields={"doc_id": "{{ input.doc_id }}"},
        )
        output = await ss_run(step, ctx, store)
        assert all(r["doc_id"] == "doc-a" for r in output["records"])
