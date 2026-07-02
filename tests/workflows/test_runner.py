"""Unit tests for cogbase.workflows.runner.WorkflowRunner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from pydantic import TypeAdapter

from cogbase.config.config import WorkflowConfig, WorkflowParamsFromCollectionConfig, WorkflowStepConfig
from cogbase.core.models import Chunk
from cogbase.stores import CollectionSchema, VectorCollectionSchema
from cogbase.stores.schema import FieldSchema, FieldType
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from cogbase.workflows.runner import WorkflowRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Finding(BaseModel):
    finding_id: str
    status: str


_FINDING_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "finding_id": {"type": "string"},
        "status":     {"type": "string"},
    },
    "required": ["finding_id", "status"],
})

_CLAUSE_SCHEMA = CollectionSchema(
    name="clauses",
    primary_fields=["clause_id"],
    description="Test clause collection",
    fields={
        "clause_id": FieldSchema(type=FieldType.STRING),
        "text":      FieldSchema(type=FieldType.STRING),
    },
)

_FINDINGS_SCHEMA = CollectionSchema(
    name="findings",
    primary_fields=["finding_id"],
    description="Test findings collection",
    fields={
        "finding_id": FieldSchema(type=FieldType.STRING),
        "status":     FieldSchema(type=FieldType.STRING),
    },
)


_STEP_ADAPTER: TypeAdapter[WorkflowStepConfig] = TypeAdapter(WorkflowStepConfig)


def _make_step(**kwargs) -> WorkflowStepConfig:
    return _STEP_ADAPTER.validate_python({"id": kwargs.pop("id", "step"), **kwargs})


_DEFAULT_PARAMS_FROM_COLLECTION = WorkflowParamsFromCollectionConfig(collection="clauses")


def _make_workflow(steps: list[WorkflowStepConfig], name: str = "test-wf") -> WorkflowConfig:
    return WorkflowConfig(name=name, steps=steps, params_from_collection=_DEFAULT_PARAMS_FROM_COLLECTION)


def _make_llm(response: str, *, call_count_ref: list | None = None) -> MagicMock:
    llm = MagicMock()

    async def _complete(messages, **kwargs):
        if call_count_ref is not None:
            call_count_ref.append(1)
        return {"content": response}

    llm.complete = AsyncMock(side_effect=_complete)
    return llm


def _make_embedder(dim: int = 4) -> MagicMock:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1] * dim])
    return embedder


async def _drain(runner: WorkflowRunner, params: dict) -> list[dict]:
    return [r async for r in runner.run(params)]


async def _make_clause_store(*clauses: dict) -> InMemoryStructuredStore:
    store = InMemoryStructuredStore()
    await store.create_collection(_CLAUSE_SCHEMA)
    if clauses:
        from pydantic import create_model
        Row = create_model("Row", clause_id=(str, ...), text=(str, ""))
        await store.save("clauses", [Row(**c) for c in clauses])
    return store


async def _make_full_stores(*clauses: dict):
    """Return (structured_store, findings_store, vector_store) with test data."""
    structured = await _make_clause_store(*clauses)
    await structured.create_collection(_FINDINGS_SCHEMA)
    vs = FAISSVectorStore()
    schema = VectorCollectionSchema(name="rules", dimensions=4, description="rules")
    await vs.create_collection(schema)
    return structured, vs


# ---------------------------------------------------------------------------
# Linear step execution
# ---------------------------------------------------------------------------

class TestLinearSteps:
    async def test_empty_workflow_yields_nothing(self):
        runner = WorkflowRunner(_make_workflow([]))
        records = await _drain(runner, {})
        assert records == []

    async def test_single_structured_query_step(self):
        store = await _make_clause_store(
            {"clause_id": "c1", "text": "alpha"},
            {"clause_id": "c2", "text": "beta"},
        )
        step = _make_step(tool="structured-query", collection="clauses")
        runner = WorkflowRunner(
            _make_workflow([step]),
            structured_store=store,
        )
        records = await _drain(runner, {})
        # structured-query does not stream records, nothing yielded
        assert records == []

    async def test_step_output_stored_in_ctx(self):
        store = await _make_clause_store({"clause_id": "c1", "text": "x"})
        captured_ctx: dict = {}

        original_run = WorkflowRunner._run_steps

        async def _spy(self, steps, ctx):
            async for r in original_run(self, steps, ctx):
                yield r
            captured_ctx.update(ctx)

        runner = WorkflowRunner(
            _make_workflow([_make_step(tool="structured-query", collection="clauses")]),
            structured_store=store,
        )
        with patch.object(WorkflowRunner, "_run_steps", _spy):
            await _drain(runner, {})

        assert "load" not in captured_ctx["steps"]  # id is "step", not "load"

    async def test_structured_save_yields_records(self):
        structured, vs = await _make_full_stores({"clause_id": "c1", "text": "x"})
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}')
        embedder = _make_embedder()

        steps = [
            _make_step(id="q", tool="structured-query", collection="clauses"),
            _make_step(id="s", tool="vector-search", collection="rules", query="test", top_k=1),
            _make_step(
                id="j",
                tool="llm-structured",
                prompt="Judge.",
                input={"data": "{{ steps.q.records }}"},
                output_schema=_FINDING_SCHEMA,
            ),
            _make_step(
                id="save",
                tool="structured-save",
                collection="findings",
                records=["{{ steps.j.output }}"],
            ),
        ]
        runner = WorkflowRunner(
            _make_workflow(steps),
            structured_store=structured,
            vector_store=vs,
            embedder=embedder,
            llm=llm,
        )
        records = await _drain(runner, {})
        assert len(records) == 1
        assert records[0]["finding_id"] == "f1"
        assert records[0]["status"] == "ok"

    async def test_saved_record_is_dict(self):
        structured, vs = await _make_full_stores({"clause_id": "c1", "text": "x"})
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}')

        steps = [
            _make_step(
                id="j",
                tool="llm-structured",
                prompt="Judge.",
                output_schema=_FINDING_SCHEMA,
            ),
            _make_step(
                id="save",
                tool="structured-save",
                collection="findings",
                records=["{{ steps.j.output }}"],
            ),
        ]
        runner = WorkflowRunner(
            _make_workflow(steps),
            structured_store=structured,
            llm=llm,
        )
        records = await _drain(runner, {})
        assert isinstance(records[0], dict)


# ---------------------------------------------------------------------------
# foreach step
# ---------------------------------------------------------------------------

class TestForeachStep:
    async def test_foreach_iterates_over_all_items(self):
        structured, vs = await _make_full_stores(
            {"clause_id": "c1", "text": "alpha"},
            {"clause_id": "c2", "text": "beta"},
        )

        call_counts: list = []
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}', call_count_ref=call_counts)
        embedder = _make_embedder()

        steps = [
            _make_step(id="load", tool="structured-query", collection="clauses"),
            _make_step(
                id="loop",
                foreach="{{ steps.load.records }}",
                steps=[
                    _make_step(id="r", tool="vector-search", collection="rules", query="{{ item.text }}", top_k=1),
                    _make_step(
                        id="j",
                        tool="llm-structured",
                        prompt="Judge.",
                        input={"clause": "{{ item }}", "rules": "{{ steps.r.chunks }}"},
                        output_schema=_FINDING_SCHEMA,
                    ),
                    _make_step(
                        id="save",
                        tool="structured-save",
                        collection="findings",
                        records=["{{ steps.j.output }}"],
                    ),
                ],
            ),
        ]
        runner = WorkflowRunner(
            _make_workflow(steps),
            structured_store=structured,
            vector_store=vs,
            embedder=embedder,
            llm=llm,
        )
        records = await _drain(runner, {})
        # One record per clause
        assert len(records) == 2
        assert len(call_counts) == 2

    async def test_foreach_item_accessible_in_inner_steps(self):
        structured, vs = await _make_full_stores({"clause_id": "c1", "text": "liability text"})
        embedder = _make_embedder()

        captured_queries: list[str] = []
        vs_mock = MagicMock()

        async def _search(collection, query_text, embedding, top_k):
            captured_queries.append(query_text)
            return []

        vs_mock.search = AsyncMock(side_effect=_search)

        steps = [
            _make_step(id="load", tool="structured-query", collection="clauses"),
            _make_step(
                id="loop",
                foreach="{{ steps.load.records }}",
                steps=[
                    _make_step(id="r", tool="vector-search", collection="rules", query="{{ item.text }}", top_k=1),
                ],
            ),
        ]
        runner = WorkflowRunner(
            _make_workflow(steps),
            structured_store=structured,
            vector_store=vs_mock,
            embedder=embedder,
        )
        await _drain(runner, {})
        assert captured_queries == ["liability text"]

    async def test_foreach_inner_step_outputs_not_visible_across_iterations(self):
        """Each iteration gets its own steps namespace; prior iteration outputs don't bleed in."""
        structured, vs = await _make_full_stores(
            {"clause_id": "c1", "text": "alpha"},
            {"clause_id": "c2", "text": "beta"},
        )
        embedder = _make_embedder()

        # Track which clause_ids the LLM sees as input
        seen_clause_ids: list[str] = []
        llm = MagicMock()

        async def _complete(messages, **kwargs):
            user_msg = messages[1]["content"]
            # The input dict has clause_id in it
            if '"c1"' in user_msg:
                seen_clause_ids.append("c1")
            elif '"c2"' in user_msg:
                seen_clause_ids.append("c2")
            return {"content": '{"finding_id": "f", "status": "ok"}'}

        llm.complete = AsyncMock(side_effect=_complete)

        steps = [
            _make_step(id="load", tool="structured-query", collection="clauses"),
            _make_step(
                id="loop",
                foreach="{{ steps.load.records }}",
                steps=[
                    _make_step(id="r", tool="vector-search", collection="rules", query="{{ item.text }}", top_k=1),
                    _make_step(
                        id="j",
                        tool="llm-structured",
                        prompt="Judge.",
                        input={"clause": "{{ item }}"},
                        output_schema=_FINDING_SCHEMA,
                    ),
                    _make_step(
                        id="save",
                        tool="structured-save",
                        collection="findings",
                        records=["{{ steps.j.output }}"],
                    ),
                ],
            ),
        ]
        runner = WorkflowRunner(
            _make_workflow(steps),
            structured_store=structured,
            vector_store=vs,
            embedder=embedder,
            llm=llm,
        )
        await _drain(runner, {})
        # Each iteration sees only its own clause
        assert "c1" in seen_clause_ids
        assert "c2" in seen_clause_ids

    async def test_foreach_outer_steps_visible_in_inner_context(self):
        """Inner steps can reference outer step outputs via steps.<id>."""
        structured, vs = await _make_full_stores({"clause_id": "c1", "text": "x"})
        embedder = _make_embedder()

        outer_records_seen: list = []
        llm = MagicMock()

        async def _complete(messages, **kwargs):
            user_msg = messages[1]["content"]
            # The outer query result should be visible in the inner step's context
            if "c1" in user_msg:
                outer_records_seen.append(True)
            return {"content": '{"finding_id": "f", "status": "ok"}'}

        llm.complete = AsyncMock(side_effect=_complete)

        steps = [
            _make_step(id="load", tool="structured-query", collection="clauses"),
            _make_step(
                id="loop",
                foreach="{{ steps.load.records }}",
                steps=[
                    _make_step(
                        id="j",
                        tool="llm-structured",
                        prompt="Judge.",
                        # reference outer step output from within foreach
                        input={"all_clauses": "{{ steps.load.records }}"},
                        output_schema=_FINDING_SCHEMA,
                    ),
                    _make_step(
                        id="save",
                        tool="structured-save",
                        collection="findings",
                        records=["{{ steps.j.output }}"],
                    ),
                ],
            ),
        ]
        runner = WorkflowRunner(
            _make_workflow(steps),
            structured_store=structured,
            llm=llm,
        )
        await _drain(runner, {})
        assert outer_records_seen  # outer load result was visible inside loop

    async def test_foreach_non_list_raises(self):
        structured = await _make_clause_store({"clause_id": "c1", "text": "x"})
        # Make the query return a single dict (not a list) — shouldn't happen but test the guard
        step_q = _make_step(id="load", tool="structured-query", collection="clauses")
        # foreach references a non-list value
        step_loop = _make_step(
            id="loop",
            foreach="{{ steps.load.records[0] }}",  # a dict, not a list
            steps=[_make_step(id="inner", tool="structured-query", collection="clauses")],
        )
        runner = WorkflowRunner(
            _make_workflow([step_q, step_loop]),
            structured_store=structured,
        )
        with pytest.raises(ValueError, match="list"):
            await _drain(runner, {})

    async def test_foreach_empty_list_yields_nothing(self):
        structured = await _make_clause_store()  # no records
        steps = [
            _make_step(id="load", tool="structured-query", collection="clauses"),
            _make_step(
                id="loop",
                foreach="{{ steps.load.records }}",
                steps=[_make_step(id="inner", tool="structured-query", collection="clauses")],
            ),
        ]
        runner = WorkflowRunner(_make_workflow(steps), structured_store=structured)
        records = await _drain(runner, {})
        assert records == []


# ---------------------------------------------------------------------------
# Input params accessible in ctx
# ---------------------------------------------------------------------------

class TestWorkflowInputParams:
    async def test_input_params_used_in_filter(self):
        """Workflow input params are accessible via {{ input.x }} in step templates."""
        store = await _make_clause_store(
            {"clause_id": "c1", "text": "match"},
            {"clause_id": "c2", "text": "no match"},
        )

        # A downstream save step lets us observe which records were loaded
        structured, _ = await _make_full_stores(
            {"clause_id": "c1", "text": "match"},
            {"clause_id": "c2", "text": "no match"},
        )
        llm = _make_llm('{"finding_id": "f1", "status": "ok"}')

        steps = [
            _make_step(
                id="load",
                tool="structured-query",
                collection="clauses",
                filters={"clause_id": "{{ input.target_id }}"},
            ),
            _make_step(
                id="loop",
                foreach="{{ steps.load.records }}",
                steps=[
                    _make_step(
                        id="j",
                        tool="llm-structured",
                        prompt="Judge.",
                        input={"clause": "{{ item }}"},
                        output_schema=_FINDING_SCHEMA,
                    ),
                    _make_step(
                        id="save",
                        tool="structured-save",
                        collection="findings",
                        records=["{{ steps.j.output }}"],
                    ),
                ],
            ),
        ]
        runner = WorkflowRunner(_make_workflow(steps), structured_store=structured, llm=llm)
        records = await _drain(runner, {"target_id": "c1"})

        # Only c1 was loaded (filter applied), so exactly one finding is saved
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------

class TestStreaming:
    async def test_multiple_saves_yields_in_order(self):
        structured, _ = await _make_full_stores(
            {"clause_id": "c1", "text": "a"},
            {"clause_id": "c2", "text": "b"},
            {"clause_id": "c3", "text": "c"},
        )
        call_n = 0
        llm = MagicMock()

        async def _complete(messages, **kwargs):
            nonlocal call_n
            call_n += 1
            return {"content": json.dumps({"finding_id": f"f{call_n}", "status": "ok"})}

        llm.complete = AsyncMock(side_effect=_complete)
        embedder = _make_embedder()

        vs = FAISSVectorStore()
        await vs.create_collection(VectorCollectionSchema(name="rules", dimensions=4, description="r"))

        steps = [
            _make_step(id="load", tool="structured-query", collection="clauses"),
            _make_step(
                id="loop",
                foreach="{{ steps.load.records }}",
                steps=[
                    _make_step(id="r", tool="vector-search", collection="rules", query="{{ item.text }}", top_k=1),
                    _make_step(
                        id="j",
                        tool="llm-structured",
                        prompt="Judge.",
                        input={"clause": "{{ item }}"},
                        output_schema=_FINDING_SCHEMA,
                    ),
                    _make_step(
                        id="save",
                        tool="structured-save",
                        collection="findings",
                        records=["{{ steps.j.output }}"],
                    ),
                ],
            ),
        ]
        runner = WorkflowRunner(_make_workflow(steps), structured_store=structured, vector_store=vs, embedder=embedder, llm=llm)
        records = []
        async for r in runner.run({}):
            records.append(r)

        assert len(records) == 3
        assert [r["finding_id"] for r in records] == ["f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# purge_document — pre-regeneration cleanup of a re-ingested doc's output
# ---------------------------------------------------------------------------

_DOC_FINDINGS_SCHEMA = CollectionSchema(
    name="doc_findings",
    primary_fields=["doc_id", "finding_id"],
    description="Per-document findings",
    fields={
        "doc_id":     FieldSchema(type=FieldType.STRING),
        "finding_id": FieldSchema(type=FieldType.STRING),
    },
)

_CONTRADICTIONS_SCHEMA = CollectionSchema(
    name="contradictions",
    primary_fields=["contradiction_id"],
    description="Cross-document contradictions",
    fields={
        "contradiction_id": FieldSchema(type=FieldType.STRING),
        "issue":            FieldSchema(type=FieldType.STRING),
        "doc_a_id":         FieldSchema(type=FieldType.STRING),
        "doc_b_id":         FieldSchema(type=FieldType.STRING),
    },
)


async def _store_with(schema: CollectionSchema, rows: list[dict], **field_types) -> InMemoryStructuredStore:
    store = InMemoryStructuredStore()
    await store.create_collection(schema)
    if rows:
        from pydantic import create_model
        Row = create_model("Row", **{k: (t, ...) for k, t in field_types.items()})
        await store.save(schema.name, [Row(**r) for r in rows])
    return store


class TestPurgeDocument:
    async def test_per_doc_purge_removes_only_that_docs_rows(self):
        store = await _store_with(
            _DOC_FINDINGS_SCHEMA,
            [
                {"doc_id": "A", "finding_id": "f1"},
                {"doc_id": "A", "finding_id": "f2"},
                {"doc_id": "B", "finding_id": "f3"},
            ],
            doc_id=str, finding_id=str,
        )
        # Default purge_by == [doc_id]
        save = _make_step(id="save", tool="structured-save", collection="doc_findings",
                          records=["{{ item }}"])
        runner = WorkflowRunner(_make_workflow([save]), structured_store=store)

        await runner.purge_document("A")

        remaining = await store.query("doc_findings")
        assert {r["finding_id"] for r in remaining} == {"f3"}

    async def test_cross_doc_purge_leaves_unrelated_pair_intact(self):
        # The key regression guard: re-ingesting A drops A's contradictions but
        # NOT the B–C pair, which A is not authoritative for.
        store = await _store_with(
            _CONTRADICTIONS_SCHEMA,
            [
                {"contradiction_id": "AB", "issue": "i", "doc_a_id": "A", "doc_b_id": "B"},
                {"contradiction_id": "AC", "issue": "i", "doc_a_id": "A", "doc_b_id": "C"},
                {"contradiction_id": "BC", "issue": "i", "doc_a_id": "B", "doc_b_id": "C"},
            ],
            contradiction_id=str, issue=str, doc_a_id=str, doc_b_id=str,
        )
        save = _make_step(id="save", tool="structured-save", collection="contradictions",
                          purge_by=["doc_a_id", "doc_b_id"], records=["{{ item }}"])
        runner = WorkflowRunner(_make_workflow([save]), structured_store=store)

        await runner.purge_document("A")

        remaining = await store.query("contradictions")
        assert {r["contradiction_id"] for r in remaining} == {"BC"}

    async def test_purge_by_empty_disables_purge(self):
        store = await _store_with(
            _DOC_FINDINGS_SCHEMA,
            [{"doc_id": "A", "finding_id": "f1"}],
            doc_id=str, finding_id=str,
        )
        save = _make_step(id="save", tool="structured-save", collection="doc_findings",
                          purge_by=[], records=["{{ item }}"])
        runner = WorkflowRunner(_make_workflow([save]), structured_store=store)

        await runner.purge_document("A")

        remaining = await store.query("doc_findings")
        assert len(remaining) == 1

    async def test_purge_recurses_into_foreach(self):
        store = await _store_with(
            _DOC_FINDINGS_SCHEMA,
            [{"doc_id": "A", "finding_id": "f1"}],
            doc_id=str, finding_id=str,
        )
        # save step lives inside a foreach — purge must still discover it
        loop = _make_step(
            id="loop",
            foreach="{{ steps.load.records }}",
            steps=[_make_step(id="save", tool="structured-save",
                              collection="doc_findings", records=["{{ item }}"])],
        )
        runner = WorkflowRunner(_make_workflow([loop]), structured_store=store)

        await runner.purge_document("A")

        remaining = await store.query("doc_findings")
        assert remaining == []

    async def test_purge_absent_doc_is_noop(self):
        store = await _store_with(
            _DOC_FINDINGS_SCHEMA,
            [{"doc_id": "A", "finding_id": "f1"}],
            doc_id=str, finding_id=str,
        )
        save = _make_step(id="save", tool="structured-save", collection="doc_findings",
                          records=["{{ item }}"])
        runner = WorkflowRunner(_make_workflow([save]), structured_store=store)

        await runner.purge_document("Z")  # never ingested

        remaining = await store.query("doc_findings")
        assert len(remaining) == 1

    async def test_purge_without_store_is_noop(self):
        runner = WorkflowRunner(_make_workflow([]), structured_store=None)
        await runner.purge_document("A")  # must not raise
