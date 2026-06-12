"""Unit tests for cogbase.core.query_runner.QueryRunner.

Tests are grouped by concern:

  select()              — skill routing LLM call
  build_system_prompt() — prompt assembly
  run() / skill mode    — skill selection + tool-call loop (python/shell)
  run() / retrieval     — structured_lookup / vector_search system tools
  _execute_tool()       — python, shell, system-tool, unknown-tool branches
  compact_messages()    — context compression
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.core.query_runner import (
    DocumentSlice,
    MemoryTiers,
    QueryResult,
    QueryRunner as Runner,
    RetrievalResources,
    _extract_cited_ids,
    _filter_cited_chunks,
    _filter_cited_slices,
)
from cogbase.llms.base import ChatMessage, CompletionResult, SystemTool
from cogbase.skills.skill import Skill


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_skill(name: str, description: str = "A skill.", markdown: str = "") -> Skill:
    return Skill(
        name=name,
        description=description,
        raw_markdown=markdown or f"# {name}\nDo stuff.",
    )


def _text_result(content: str) -> CompletionResult:
    return {"content": content, "tool_calls": None}


def _tool_result(name: str, arguments: dict, call_id: str = "call-1") -> CompletionResult:
    return {
        "content": None,
        "tool_calls": [{"id": call_id, "name": name, "arguments": json.dumps(arguments)}],
    }


def _make_llm(*results: CompletionResult) -> MagicMock:
    llm = MagicMock()
    queue = list(results)
    pos = [0]

    async def _stream_gen(result: CompletionResult):
        if result.get("content"):
            yield result["content"]
        if result.get("tool_calls"):
            yield result

    def _pop():
        r = queue[pos[0]]
        pos[0] += 1
        return r

    llm.complete = AsyncMock(side_effect=lambda *a, **kw: _pop())
    llm.complete_stream = MagicMock(side_effect=lambda *a, **kw: _stream_gen(_pop()))
    # A real context window; otherwise MagicMock's int() defaults to 1, collapsing
    # the summariser's chunk budget to 1 token and fanning the transcript into
    # many single-token chunks (draining the queued responses above).
    llm.context_window = MagicMock(return_value=128_000)
    return llm


def _str_chunks(chunks: list) -> list[str]:
    return [c for c in chunks if isinstance(c, str)]


def _result_with_usage(
    content: str | None = None,
    tool_calls=None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> CompletionResult:
    return {
        "content": content,
        "tool_calls": tool_calls,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _make_llm_tracking(*results: CompletionResult) -> MagicMock:
    """Like _make_llm but always yields the final CompletionResult dict so usage is captured."""
    llm = MagicMock()
    queue = list(results)
    pos = [0]

    async def _stream_gen(result: CompletionResult):
        if result.get("content"):
            yield result["content"]
        yield result  # always emit the dict so the runner can read usage

    def _pop():
        r = queue[pos[0]]
        pos[0] += 1
        return r

    llm.complete = AsyncMock(side_effect=lambda *a, **kw: _pop())
    llm.complete_stream = MagicMock(side_effect=lambda *a, **kw: _stream_gen(_pop()))
    # See _make_llm: give the summariser a sane chunk budget.
    llm.context_window = MagicMock(return_value=128_000)
    return llm


def _make_document_store(docs: dict[str, str]) -> MagicMock:
    store = MagicMock()
    async def _load(collection, doc_id):
        if doc_id not in docs:
            raise KeyError(doc_id)
        return docs[doc_id]
    store.load = AsyncMock(side_effect=_load)
    return store


def _doc_store() -> MagicMock:
    """Minimal document store mock for tests that don't exercise document reading."""
    return _make_document_store({})


def _make_runner(
    app_id,
    llm,
    document_store=None,
    *,
    structured_store=None,
    vector_store=None,
    embedder=None,
    structured_schemas=None,
    vector_schemas=None,
    short_term=None,
    episodic=None,
    long_term=None,
    **kwargs,
) -> Runner:
    """Build a QueryRunner from flat kwargs, bundling into the dependency
    dataclasses so tests stay terse against the bundled constructor."""
    return Runner(
        app_id,
        llm,
        RetrievalResources(
            document_store=document_store if document_store is not None else _doc_store(),
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            structured_schemas=structured_schemas,
            vector_schemas=vector_schemas,
        ),
        MemoryTiers(short_term=short_term, episodic=episodic, long_term=long_term),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_returns_matching_skill():
    skills = [_make_skill("weather"), _make_skill("model-usage")]
    llm = _make_llm(_text_result("weather"))
    runner = _make_runner("test", llm, _doc_store(), skills=skills)
    result = await runner.select("What's the weather?")
    assert result is skills[0]


@pytest.mark.asyncio
async def test_select_returns_none_for_no_match():
    skills = [_make_skill("weather")]
    llm = _make_llm(_text_result("none"))
    runner = _make_runner("test", llm, _doc_store(), skills=skills)
    result = await runner.select("Tell me a joke")
    assert result is None


@pytest.mark.asyncio
async def test_select_returns_none_for_unknown_skill_name():
    skills = [_make_skill("weather")]
    llm = _make_llm(_text_result("nonexistent"))
    runner = _make_runner("test", llm, _doc_store(), skills=skills)
    result = await runner.select("something")
    assert result is None


@pytest.mark.asyncio
async def test_select_empty_skills_returns_none_without_llm_call():
    llm = MagicMock()
    llm.complete = AsyncMock()
    runner = _make_runner("test", llm, _doc_store())
    result = await runner.select("anything")
    assert result is None
    llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# build_system_prompt()
# ---------------------------------------------------------------------------

def test_build_system_prompt_includes_skill_markdown():
    skill = _make_skill("weather", markdown="# Weather\nRun curl.")
    runner = _make_runner("test", MagicMock(), _doc_store())
    prompt = runner.build_system_prompt("You are helpful.", skill)
    assert "# Weather\nRun curl." in prompt
    assert "Active Skill: weather" in prompt



def test_build_system_prompt_includes_metadata():
    skill = _make_skill("weather")
    skill.metadata = {"requires": {"bins": ["curl"]}}
    runner = _make_runner("test", MagicMock(), _doc_store())
    prompt = runner.build_system_prompt("base", skill)
    assert "curl" in prompt


# ---------------------------------------------------------------------------
# run() — skill mode, happy paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_skill_no_tools_yields_answer_and_result():
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("weather"),      # select
        _text_result("It is sunny."), # answer
    )
    runner = _make_runner("test", llm, _doc_store(), skills=skills)
    chunks = [c async for c in runner.run("Weather?")]
    assert any("Using skill: weather" in c for c in _str_chunks(chunks))
    assert _str_chunks(chunks)[-1] == "It is sunny."
    assert isinstance(chunks[-1], QueryResult)
    assert chunks[-1].answer == "It is sunny.\n"


@pytest.mark.asyncio
async def test_run_skill_single_tool_call_then_answer():
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("weather"),                                 # select
        _tool_result("shell", {"command": "curl wttr.in/NYC"}), # tool call
        _text_result("The weather in NYC is 72°F."),            # answer
    )
    runner = _make_runner("test", llm, _doc_store(), skills=skills)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="72°F, sunny")):
        chunks = [c async for c in runner.run("Weather in NYC?")]
    assert any("Executing: shell" in c for c in _str_chunks(chunks))
    assert _str_chunks(chunks)[-1] == "The weather in NYC is 72°F."


@pytest.mark.asyncio
async def test_run_skill_emits_status_once():
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("weather"),                             # select
        _tool_result("shell", {"command": "curl wttr.in"}), # tool call
        _text_result("Sunny."),                              # answer
    )
    runner = _make_runner("test", llm, _doc_store(), skills=skills)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="sunny")):
        chunks = [c async for c in runner.run("Weather?")]
    assert sum(1 for c in _str_chunks(chunks) if "Using skill" in c) == 1


@pytest.mark.asyncio
async def test_run_no_skill_selected_answers_directly():
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("none"),          # select → no skill
        _text_result("I don't know."), # direct answer
    )
    runner = _make_runner("test", llm, _doc_store(), skills=skills)
    chunks = [c async for c in runner.run("What is 2+2?")]
    assert _str_chunks(chunks)[-1] == "I don't know."
    assert not any("Using skill" in c for c in _str_chunks(chunks))
    assert isinstance(chunks[-1], QueryResult)


@pytest.mark.asyncio
async def test_run_max_calls_exceeded_yields_error():
    skills = [_make_skill("weather")]
    tool = _tool_result("shell", {"command": "echo hi"})
    llm = _make_llm(
        _text_result("weather"), tool, tool, # select + 2 tool rounds
    )
    runner = _make_runner("test", llm, _doc_store(), max_calls=2, skills=skills)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="ok")):
        chunks = [c async for c in runner.run("Weather?")]
    assert any("unable to complete" in c.answer.lower() for c in chunks if isinstance(c, QueryResult))
    assert isinstance(chunks[-1], QueryResult)


# ---------------------------------------------------------------------------
# run() — retrieval mode (no skills, stores configured)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_retrieval_direct_answer():
    """No skills, no tool calls — LLM answers directly."""
    llm = _make_llm(_text_result("The answer is 42."))
    runner = _make_runner("test", llm, _doc_store())
    chunks = [c async for c in runner.run("What is the answer?")]
    assert _str_chunks(chunks)[-1] == "The answer is 42."
    assert isinstance(chunks[-1], QueryResult)
    assert chunks[-1].answer == "The answer is 42.\n"


@pytest.mark.asyncio
async def test_run_retrieval_structured_lookup_populates_records():
    """structured_lookup results are accumulated in QueryResult.structured_records."""
    from pydantic import BaseModel as PydanticModel
    from cogbase.stores import CollectionSchema, FieldSchema, FieldType
    from cogbase.stores.structured.memory import InMemoryStructuredStore

    class Fact(PydanticModel):
        title: str

    store = InMemoryStructuredStore()
    schema = CollectionSchema(
        name="facts",
        description="Test facts collection.",
        primary_fields=["title"],
        fields={"title": FieldSchema(type=FieldType.STRING)},
    )
    await store.create_collection(schema)
    await store.save("facts", [Fact(title="Foo"), Fact(title="Bar")])

    llm = _make_llm(
        _tool_result("structured_lookup", {"collection": "facts"}),
        _text_result("Found: Foo, Bar."),
    )
    runner = _make_runner("test", llm, _doc_store(), structured_store=store)
    chunks = [c async for c in runner.run("list all facts")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert len(result.structured_records) == 2
    assert result.passthrough is False
    assert result.answer == "Found: Foo, Bar.\n"


@pytest.mark.asyncio
async def test_run_retrieval_passthrough_when_records_exceed_threshold():
    """structured_lookup with large result bypasses LLM synthesis."""
    from pydantic import BaseModel as PydanticModel
    from cogbase.stores import CollectionSchema, FieldSchema, FieldType
    from cogbase.stores.structured.memory import InMemoryStructuredStore

    class BigRecord(PydanticModel):
        data: str

    store = InMemoryStructuredStore()
    schema = CollectionSchema(
        name="big",
        description="Test collection with many records.",
        primary_fields=["data"],
        fields={"data": FieldSchema(type=FieldType.STRING)},
    )
    await store.create_collection(schema)
    # Each record ~25 chars × 400 records ≈ 10 000 chars ≈ 2 500 tokens > default 2 000
    await store.save("big", [BigRecord(data="x" * 25) for _ in range(400)])

    llm = _make_llm(
        _tool_result("structured_lookup", {"collection": "big"}),
    )
    runner = _make_runner("test", llm, _doc_store(), structured_store=store, passthrough_token_threshold=2000)
    chunks = [c async for c in runner.run("dump big")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert result.passthrough is True
    assert len(result.structured_records) == 400
    # LLM should NOT have been called for synthesis (only the one tool-call completion)
    assert llm.complete_stream.call_count == 1


@pytest.mark.asyncio
async def test_run_retrieval_vector_search_populates_chunks():
    """vector_search results are accumulated in QueryResult.chunks."""
    from cogbase.core.models import Chunk
    from cogbase.embeddings.base import EmbeddingBase
    from cogbase.stores import VectorStoreBase

    class _FakeEmbedder(EmbeddingBase):
        async def embed(self, texts):
            return [[0.1] * 4 for _ in texts]

    class _FakeVectorStore(VectorStoreBase):
        async def upsert(self, collection, chunks): pass
        async def search(self, collection, query_text, embedding, top_k):
            return [Chunk(chunk_id="c1", doc_id="d1", text="relevant passage", embedding=[0.1]*4)]
        async def delete(self, collection, doc_id): pass
        async def delete_collection(self, collection): pass
        async def create_collection(self, schema): pass

    llm = _make_llm(
        _tool_result("vector_search", {"query": "relevant", "collection": "docs"}),
        _text_result("Here is the relevant passage."),
    )
    runner = _make_runner(
        "test",
        llm,
        _doc_store(),
        vector_store=_FakeVectorStore(),
        embedder=_FakeEmbedder(),
    )
    chunks = [c async for c in runner.run("find relevant")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "relevant passage"


# ---------------------------------------------------------------------------
# run() — retrieval tool availability (tool_defs introspection)
# ---------------------------------------------------------------------------

def test_tool_defs_structured_only():
    from cogbase.stores import CollectionSchema, FieldSchema, FieldType
    from cogbase.stores.structured.memory import InMemoryStructuredStore
    schema = CollectionSchema(
        name="facts",
        description="desc",
        primary_fields=["id"],
        fields={"id": FieldSchema(type=FieldType.STRING)},
    )
    runner = _make_runner(
        "test", MagicMock(), _doc_store(),
        structured_store=InMemoryStructuredStore(),
        structured_schemas=[schema],
    )
    names = [t["name"] for t in runner._tool_defs]
    assert "structured_lookup" in names
    assert "vector_search" not in names


def test_tool_defs_vector_only():
    from cogbase.embeddings.base import EmbeddingBase
    from cogbase.stores import VectorCollectionSchema, VectorStoreBase

    class _V(VectorStoreBase):
        async def upsert(self, c, chunks): pass
        async def search(self, c, e, k): return []
        async def delete(self, c, d): pass
        async def delete_collection(self, c): pass
        async def create_collection(self, s): pass

    class _E(EmbeddingBase):
        async def embed(self, texts): return [[0.0]]

    schema = VectorCollectionSchema(name="docs", dimensions=4, description="desc")
    runner = _make_runner(
        "test", MagicMock(), _doc_store(),
        vector_store=_V(),
        embedder=_E(),
        vector_schemas=[schema],
    )
    names = [t["name"] for t in runner._tool_defs]
    assert "vector_search" in names
    assert "structured_lookup" not in names


def test_tool_defs_no_schemas_has_only_read_document():
    runner = _make_runner("test", MagicMock(), _doc_store())
    names = [t["name"] for t in runner._tool_defs]
    assert "read_document" in names
    assert "structured_lookup" not in names
    assert "vector_search" not in names


# ---------------------------------------------------------------------------
# run() — custom system tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_custom_system_tool_is_called():
    """A custom SystemTool registered at init is dispatched correctly."""
    called_with: list[dict] = []

    async def _handler(inputs: dict) -> str:
        called_with.append(inputs)
        return "custom result"

    tool_def = {
        "name": "my_tool",
        "description": "A custom tool.",
        "parameters": {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
            "required": ["arg"],
            "additionalProperties": False,
        },
    }
    system_tool = SystemTool(definition=tool_def, handler=_handler)

    llm = _make_llm(
        _tool_result("my_tool", {"arg": "hello"}),
        _text_result("Done."),
    )
    runner = _make_runner("test", llm, _doc_store(), system_tools=[system_tool])
    chunks = [c async for c in runner.run("run my tool")]
    assert called_with == [{"arg": "hello"}]
    assert _str_chunks(chunks)[-1] == "Done."


# ---------------------------------------------------------------------------
# _execute_tool()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_tool_python_returns_stdout():
    runner = _make_runner("test", MagicMock(), _doc_store())
    output = await runner._execute_tool("python", {"code": "print('hello')"})
    assert output == "hello"


@pytest.mark.asyncio
async def test_execute_tool_shell_returns_stdout():
    runner = _make_runner("test", MagicMock(), _doc_store())
    output = await runner._execute_tool("shell", {"command": "echo hi"})
    assert output == "hi"


@pytest.mark.asyncio
async def test_execute_tool_unknown_returns_error():
    runner = _make_runner("test", MagicMock(), _doc_store())
    output = await runner._execute_tool("nonexistent", {})
    assert "Unknown tool" in output


@pytest.mark.asyncio
async def test_execute_tool_python_bad_code_returns_stderr():
    runner = _make_runner("test", MagicMock(), _doc_store())
    output = await runner._execute_tool("python", {"code": "raise ValueError('boom')"})
    assert output  # stderr is captured, not empty


@pytest.mark.asyncio
async def test_execute_tool_system_tool_error_returns_message():
    def _bad_handler(inputs):
        raise RuntimeError("exploded")

    tool_def = {
        "name": "boom",
        "description": "fails",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }
    system_tool = SystemTool(definition=tool_def, handler=_bad_handler)
    runner = _make_runner("test", MagicMock(), _doc_store(), system_tools=[system_tool])
    output = await runner._execute_tool("boom", {})
    assert "Tool error" in output
    assert "exploded" in output


# ---------------------------------------------------------------------------
# run() — read_document tool
# ---------------------------------------------------------------------------

def test_tool_defs_read_document_when_document_store_and_app_name_set():
    store = _make_document_store({})
    runner = _make_runner("myapp", MagicMock(), store)
    names = [t["name"] for t in runner._tool_defs]
    assert "read_document" in names


@pytest.mark.asyncio
async def test_run_read_document_returns_slice():
    text = "A" * 100 + "B" * 100 + "C" * 100
    store = _make_document_store({"doc-1": text})
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "doc-1", "offset": 50, "length": 100}),
        _text_result("Got the slice."),
    )
    runner = _make_runner("myapp", llm, store)
    chunks = [c async for c in runner.run("read doc-1")]
    tool_output = [m["content"] for m in []]  # inspected via store.load call count
    store.load.assert_called_once_with("myapp", "doc-1")
    assert _str_chunks(chunks)[-1] == "Got the slice."


@pytest.mark.asyncio
async def test_run_read_document_slice_content_correct():
    text = "hello " * 50  # 300 chars
    store = _make_document_store({"doc-1": text})

    captured_tool_outputs: list[str] = []

    original_run = Runner._run_read_document

    async def _capturing(self, inputs):
        result = await original_run(self, inputs)
        captured_tool_outputs.append(result)
        return result

    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "doc-1", "offset": 0, "length": 12}),
        _text_result("Done."),
    )
    runner = _make_runner("myapp", llm, store)
    with patch.object(Runner, "_run_read_document", _capturing):
        [c async for c in runner.run("read doc")]

    assert len(captured_tool_outputs) == 1
    doc_slice, tool_text = captured_tool_outputs[0]
    assert isinstance(doc_slice, DocumentSlice)
    assert "hello hello" in tool_text
    assert "chars 0–12 of 300" in tool_text


@pytest.mark.asyncio
async def test_run_read_document_missing_doc_returns_error():
    store = _make_document_store({})
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "missing"}),
        _text_result("Could not find it."),
    )
    runner = _make_runner("myapp", llm, store)
    chunks = [c async for c in runner.run("read missing")]
    # The tool error message is passed back to the LLM; loop completes normally.
    assert _str_chunks(chunks)[-1] == "Could not find it."


@pytest.mark.asyncio
async def test_run_read_document_length_capped_at_10000():
    text = "x" * 20000
    store = _make_document_store({"big": text})

    captured: list[str] = []
    original = Runner._run_read_document

    async def _cap(self, inputs):
        result = await original(self, inputs)
        captured.append(result)
        return result

    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "big", "length": 99999}),
        _text_result("Done."),
    )
    runner = _make_runner("myapp", llm, store)
    with patch.object(Runner, "_run_read_document", _cap):
        [c async for c in runner.run("read big")]

    assert len(captured) == 1
    doc_slice, tool_text = captured[0]
    assert isinstance(doc_slice, DocumentSlice)
    assert "chars 0–10000 of 20000" in tool_text


@pytest.mark.asyncio
async def test_run_read_document_populates_document_slices():
    """Successful read_document calls are accumulated in QueryResult.document_slices."""
    text = "X" * 500
    store = _make_document_store({"doc-1": text})
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "doc-1", "offset": 10, "length": 50}),
        _text_result("Done."),
    )
    runner = _make_runner("myapp", llm, store)
    chunks = [c async for c in runner.run("read doc-1")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert len(result.document_slices) == 1
    s = result.document_slices[0]
    assert isinstance(s, DocumentSlice)
    assert s.doc_id == "doc-1"
    assert s.offset == 10
    assert s.length == 50
    assert s.text == "X" * 50


@pytest.mark.asyncio
async def test_run_read_document_missing_doc_has_empty_document_slices():
    """A failed read_document (doc not found) does not populate document_slices."""
    store = _make_document_store({})
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "missing"}),
        _text_result("Could not find it."),
    )
    runner = _make_runner("myapp", llm, store)
    chunks = [c async for c in runner.run("read missing")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert result.document_slices == []


# ---------------------------------------------------------------------------
# compact_messages()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_messages_returns_two_messages():
    llm = _make_llm(_text_result("Summary: did X then Y."))
    runner = _make_runner("test", llm, _doc_store())
    history: list[ChatMessage] = [
        {"role": "user", "content": "step 1"},
        {"role": "assistant", "content": "done step 1"},
    ]
    compacted = await runner.compact_messages("You are helpful.", history)
    assert len(compacted) == 2
    assert compacted[0]["role"] == "system"
    assert "Summary" in compacted[1]["content"]


# ---------------------------------------------------------------------------
# _format_chunks() — chunk label includes chunk_id
# ---------------------------------------------------------------------------

def test_format_chunks_includes_chunk_id():
    from cogbase.core.query_runner import _format_chunks
    from cogbase.core.models import Chunk

    chunk = Chunk(chunk_id="contract_001_0", doc_id="contract_001", text="Payment terms are net 30.")
    output = _format_chunks([chunk])
    assert "[contract_001_0]" in output
    assert "doc: contract_001" in output
    assert "Payment terms are net 30." in output


def test_format_chunks_chunk_id_is_bracket_key():
    from cogbase.core.query_runner import _format_chunks
    from cogbase.core.models import Chunk

    chunk = Chunk(chunk_id="c_42", doc_id="d1", text="x")
    output = _format_chunks([chunk])
    # The bracket key must be the chunk_id, not a numeric counter.
    assert "[c_42]" in output
    assert "[1]" not in output


def test_format_chunks_includes_char_offsets_when_present():
    from cogbase.core.query_runner import _format_chunks
    from cogbase.core.models import Chunk

    chunk = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="text", char_offset=100, char_length=50)
    output = _format_chunks([chunk])
    assert "chars 100–150" in output


def test_format_chunks_empty_returns_placeholder():
    from cogbase.core.query_runner import _format_chunks

    assert _format_chunks([]) == "(no passages found)"


def test_format_chunks_same_doc_sorted_by_char_offset():
    from cogbase.core.query_runner import _format_chunks
    from cogbase.core.models import Chunk

    # c2 is more relevant (returned first) but appears later in the document.
    c1 = Chunk(chunk_id="d_1", doc_id="d", text="later", char_offset=500, char_length=10)
    c2 = Chunk(chunk_id="d_0", doc_id="d", text="earlier", char_offset=100, char_length=10)
    output = _format_chunks([c1, c2])
    assert output.index("[d_0]") < output.index("[d_1]")


def test_format_chunks_cross_doc_relevance_order_preserved():
    from cogbase.core.query_runner import _format_chunks
    from cogbase.core.models import Chunk

    # doc_b is more relevant (listed first); its chunks should appear before doc_a.
    b = Chunk(chunk_id="b_0", doc_id="doc_b", text="from b", char_offset=0, char_length=6)
    a = Chunk(chunk_id="a_0", doc_id="doc_a", text="from a", char_offset=0, char_length=6)
    output = _format_chunks([b, a])
    assert output.index("[b_0]") < output.index("[a_0]")


def test_format_chunks_no_char_offset_sorts_after_offset_chunks():
    from cogbase.core.query_runner import _format_chunks
    from cogbase.core.models import Chunk

    with_offset = Chunk(chunk_id="d_0", doc_id="d", text="has offset", char_offset=200, char_length=10)
    no_offset   = Chunk(chunk_id="d_1", doc_id="d", text="no offset")
    output = _format_chunks([no_offset, with_offset])
    assert output.index("[d_0]") < output.index("[d_1]")


# ---------------------------------------------------------------------------
# _run_vector_search() — exclude_ids
# ---------------------------------------------------------------------------

def _fake_vector_store_tracking(return_chunks):
    """Returns a vector store that records the top_k it was called with."""
    from cogbase.stores import VectorStoreBase

    class _Store(VectorStoreBase):
        called_top_k: list[int] = []

        async def upsert(self, collection, chunks): pass

        async def search(self, collection, query_text, embedding, top_k):
            _Store.called_top_k.append(top_k)
            return return_chunks[:top_k]

        async def delete(self, collection, doc_id): pass
        async def delete_collection(self, collection): pass
        async def create_collection(self, schema): pass

    return _Store()


@pytest.mark.asyncio
async def test_run_vector_search_exclude_ids_filters_seen_chunks():
    """Chunks whose chunk_id is in exclude_ids are dropped from results."""
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="seen",  doc_id="d", text="already seen", embedding=[0.1] * 4)
    c2 = Chunk(chunk_id="fresh", doc_id="d", text="new result",   embedding=[0.1] * 4)

    store = _fake_vector_store_tracking([c1, c2])
    embedder = _fake_embedder()
    runner = _make_runner("test", MagicMock(), _doc_store(), vector_store=store, embedder=embedder)

    chunks, _ = await runner._run_vector_search(
        {"query": "q", "collection": "docs", "top_k": 5},
        exclude_ids={"seen"},
    )
    ids = [c.chunk_id for c in chunks]
    assert "seen" not in ids
    assert "fresh" in ids


@pytest.mark.asyncio
async def test_run_vector_search_exclude_ids_expands_search_top_k():
    """search() is called with top_k + len(exclude_ids) to compensate for filtered results."""
    from cogbase.core.models import Chunk

    chunks = [Chunk(chunk_id=f"c{i}", doc_id="d", text=f"t{i}", embedding=[0.1] * 4) for i in range(10)]
    store = _fake_vector_store_tracking(chunks)
    embedder = _fake_embedder()
    runner = _make_runner("test", MagicMock(), _doc_store(), vector_store=store, embedder=embedder)
    # Reset class-level tracker
    type(store).called_top_k = []

    await runner._run_vector_search(
        {"query": "q", "collection": "docs", "top_k": 3},
        exclude_ids={"c0", "c1"},
    )
    assert store.called_top_k[-1] == 5  # 3 + 2 excluded


@pytest.mark.asyncio
async def test_run_vector_search_no_exclude_ids_returns_full_results():
    """Without exclude_ids, all results up to top_k are returned."""
    from cogbase.core.models import Chunk

    chunks = [Chunk(chunk_id=f"c{i}", doc_id="d", text=f"t{i}", embedding=[0.1] * 4) for i in range(5)]
    store = _fake_vector_store_tracking(chunks)
    embedder = _fake_embedder()
    runner = _make_runner("test", MagicMock(), _doc_store(), vector_store=store, embedder=embedder)

    result_chunks, _ = await runner._run_vector_search(
        {"query": "q", "collection": "docs", "top_k": 5},
    )
    assert len(result_chunks) == 5


@pytest.mark.asyncio
async def test_run_retrieval_second_vector_search_skips_first_results():
    """Integration: a second vector_search call in one run() skips chunks from the first."""
    from cogbase.core.models import Chunk

    seen_chunk = Chunk(chunk_id="first",  doc_id="d", text="first call chunk",  embedding=[0.1] * 4)
    new_chunk  = Chunk(chunk_id="second", doc_id="d", text="second call chunk", embedding=[0.1] * 4)

    call_count = [0]

    class _SequentialStore:
        async def upsert(self, c, chunks): pass
        async def search(self, c, qt, emb, top_k):
            call_count[0] += 1
            if call_count[0] == 1:
                return [seen_chunk]
            # Second call: return both so we can verify seen_chunk is excluded.
            return [seen_chunk, new_chunk]
        async def delete(self, c, d): pass
        async def delete_collection(self, c): pass
        async def create_collection(self, s): pass

    llm = _make_llm(
        _tool_result("vector_search", {"query": "q1", "collection": "docs"}, call_id="v1"),
        _tool_result("vector_search", {"query": "q2", "collection": "docs"}, call_id="v2"),
        _text_result("Done."),
    )
    runner = _make_runner(
        "test",
        llm,
        _doc_store(),
        vector_store=_SequentialStore(),
        embedder=_fake_embedder(),
    )
    output = [c async for c in runner.run("search twice")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    chunk_ids = [c.chunk_id for c in result.chunks]
    # "first" may appear in final result (it was cited or fallback), but "second" must appear too
    # and "first" must not be duplicated.
    assert chunk_ids.count("first") <= 1


# ---------------------------------------------------------------------------
# _extract_cited_ids()
# ---------------------------------------------------------------------------

def test_extract_cited_ids_returns_all_bracket_ids():
    ids = _extract_cited_ids("Based on [doc_0_0] and [report.pdf:0:100], see also [doc_1_0].")
    assert ids == {"doc_0_0", "report.pdf:0:100", "doc_1_0"}


def test_extract_cited_ids_returns_empty_set_when_no_brackets():
    assert _extract_cited_ids("No citations here.") == set()


def test_extract_cited_ids_deduplicates():
    ids = _extract_cited_ids("[doc_0_0] is important. Also see [doc_0_0] again.")
    assert ids == {"doc_0_0"}


# ---------------------------------------------------------------------------
# _filter_cited_chunks()
# ---------------------------------------------------------------------------

def test_filter_cited_chunks_returns_cited_subset():
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="alpha")
    c2 = Chunk(chunk_id="doc_0_1", doc_id="doc_0", text="beta")
    c3 = Chunk(chunk_id="doc_1_0", doc_id="doc_1", text="gamma")

    cited = _extract_cited_ids("Based on [doc_0_0] and [doc_1_0], the answer is clear.")
    result = _filter_cited_chunks([c1, c2, c3], cited)
    assert result == [c1, c3]


def test_filter_cited_chunks_fallback_when_no_citations():
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="alpha")
    c2 = Chunk(chunk_id="doc_0_1", doc_id="doc_0", text="beta")

    cited = _extract_cited_ids("Based on the documents, here is the answer.")
    result = _filter_cited_chunks([c1, c2], cited)
    assert result == [c1, c2]


def test_filter_cited_chunks_returns_empty_when_cited_ids_has_only_slice_ids():
    """Bug fix: chunk filter returns [] when answer cites slice IDs, not all chunks."""
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="alpha")
    c2 = Chunk(chunk_id="doc_0_1", doc_id="doc_0", text="beta")

    # cited_ids contains a slice ID, not a chunk ID
    cited = {"report.pdf:0:100"}
    result = _filter_cited_chunks([c1, c2], cited)
    assert result == []


def test_filter_cited_chunks_empty_input_returns_empty():
    cited = _extract_cited_ids("any answer with [doc_0_0]")
    assert _filter_cited_chunks([], cited) == []


def test_filter_cited_chunks_preserves_all_chunks_order():
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="first")
    c2 = Chunk(chunk_id="doc_0_1", doc_id="doc_0", text="second")
    c3 = Chunk(chunk_id="doc_0_2", doc_id="doc_0", text="third")

    # LLM cited in reverse order — result should follow all_chunks order.
    cited = _extract_cited_ids("See [doc_0_2] and [doc_0_0].")
    result = _filter_cited_chunks([c1, c2, c3], cited)
    assert result == [c1, c3]


# ---------------------------------------------------------------------------
# run() — cited-chunk filtering integration
# ---------------------------------------------------------------------------

def _fake_vector_store_with_chunks(return_chunks):
    from cogbase.stores import VectorStoreBase

    class _Store(VectorStoreBase):
        async def upsert(self, collection, chunks): pass
        async def search(self, collection, query_text, embedding, top_k):
            return return_chunks
        async def delete(self, collection, doc_id): pass
        async def delete_collection(self, collection): pass
        async def create_collection(self, schema): pass

    return _Store()


def _fake_embedder():
    from cogbase.embeddings.base import EmbeddingBase

    class _Embedder(EmbeddingBase):
        async def embed(self, texts):
            return [[0.1] * 4 for _ in texts]

    return _Embedder()


@pytest.mark.asyncio
async def test_run_vector_search_result_contains_only_cited_chunks():
    """QueryResult.chunks is filtered to only chunks the LLM cited by chunk_id."""
    from cogbase.core.models import Chunk as ModelChunk

    store_chunks = [
        ModelChunk(chunk_id="doc_0", doc_id="d1", text="passage A", embedding=[0.1] * 4),
        ModelChunk(chunk_id="doc_1", doc_id="d1", text="passage B", embedding=[0.1] * 4),
    ]
    llm = _make_llm(
        _tool_result("vector_search", {"query": "q", "collection": "docs"}),
        _text_result("Based on [doc_0], the answer is passage A."),
    )
    runner = _make_runner(
        "test",
        llm,
        _doc_store(),
        vector_store=_fake_vector_store_with_chunks(store_chunks),
        embedder=_fake_embedder(),
    )
    output = [c async for c in runner.run("find passage")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "doc_0"


@pytest.mark.asyncio
async def test_run_vector_search_fallback_all_chunks_when_llm_cites_none():
    """QueryResult.chunks falls back to all chunks when the LLM uses no [chunk_id] citations."""
    from cogbase.core.models import Chunk as ModelChunk

    store_chunks = [
        ModelChunk(chunk_id="doc_0", doc_id="d1", text="passage A", embedding=[0.1] * 4),
        ModelChunk(chunk_id="doc_1", doc_id="d1", text="passage B", embedding=[0.1] * 4),
    ]
    llm = _make_llm(
        _tool_result("vector_search", {"query": "q", "collection": "docs"}),
        _text_result("Based on the retrieved documents, passage A is relevant."),
    )
    runner = _make_runner(
        "test",
        llm,
        _doc_store(),
        vector_store=_fake_vector_store_with_chunks(store_chunks),
        embedder=_fake_embedder(),
    )
    output = [c async for c in runner.run("find passage")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    assert len(result.chunks) == 2


# ---------------------------------------------------------------------------
# DocumentSlice.slice_id
# ---------------------------------------------------------------------------

def test_document_slice_slice_id():
    s = DocumentSlice(doc_id="report.pdf", offset=100, length=500, text="x")
    assert s.slice_id == "report.pdf:100:500"


def test_document_slice_slice_id_zero_offset():
    s = DocumentSlice(doc_id="doc", offset=0, length=2000, text="y")
    assert s.slice_id == "doc:0:2000"


# ---------------------------------------------------------------------------
# _filter_cited_slices()
# ---------------------------------------------------------------------------

def test_filter_cited_slices_returns_cited_subset():
    s1 = DocumentSlice(doc_id="a.pdf", offset=0,   length=100, text="alpha")
    s2 = DocumentSlice(doc_id="a.pdf", offset=100, length=100, text="beta")
    s3 = DocumentSlice(doc_id="b.pdf", offset=0,   length=200, text="gamma")

    cited = _extract_cited_ids(f"Based on [{s1.slice_id}] and [{s3.slice_id}], the answer is clear.")
    result = _filter_cited_slices([s1, s2, s3], cited)
    assert result == [s1, s3]


def test_filter_cited_slices_fallback_when_no_citations():
    s1 = DocumentSlice(doc_id="x.pdf", offset=0, length=50, text="alpha")
    s2 = DocumentSlice(doc_id="x.pdf", offset=50, length=50, text="beta")

    cited = _extract_cited_ids("Based on the documents, here is the answer.")
    result = _filter_cited_slices([s1, s2], cited)
    assert result == [s1, s2]


def test_filter_cited_slices_returns_empty_when_cited_ids_has_only_chunk_ids():
    """Bug fix: slice filter returns [] when answer cites chunk IDs, not all slices."""
    s1 = DocumentSlice(doc_id="x.pdf", offset=0, length=50, text="alpha")
    s2 = DocumentSlice(doc_id="x.pdf", offset=50, length=50, text="beta")

    # cited_ids contains chunk IDs, not slice IDs
    cited = {"doc_0_0", "doc_1_0"}
    result = _filter_cited_slices([s1, s2], cited)
    assert result == []


def test_filter_cited_slices_empty_input_returns_empty():
    cited = _extract_cited_ids("anything [x.pdf:0:100]")
    assert _filter_cited_slices([], cited) == []


def test_filter_cited_slices_preserves_all_slices_order():
    s1 = DocumentSlice(doc_id="d.pdf", offset=0,   length=100, text="first")
    s2 = DocumentSlice(doc_id="d.pdf", offset=100, length=100, text="second")
    s3 = DocumentSlice(doc_id="d.pdf", offset=200, length=100, text="third")

    # LLM cited in reverse order — result should follow all_slices order.
    cited = _extract_cited_ids(f"See [{s3.slice_id}] and [{s1.slice_id}].")
    result = _filter_cited_slices([s1, s2, s3], cited)
    assert result == [s1, s3]


# ---------------------------------------------------------------------------
# _run_read_document output format
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_read_document_output_uses_passage_header_and_slice_id():
    """Tool output uses 'Passage:' header with [slice_id] as the citable bracket key."""
    text = "hello world " * 20  # 240 chars
    store = _make_document_store({"doc.pdf": text})

    captured: list = []
    original = Runner._run_read_document

    async def _cap(self, inputs):
        result = await original(self, inputs)
        captured.append(result)
        return result

    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "doc.pdf", "offset": 0, "length": 24}),
        _text_result("Done."),
    )
    runner = _make_runner("myapp", llm, store)
    with patch.object(Runner, "_run_read_document", _cap):
        [c async for c in runner.run("read doc")]

    doc_slice, tool_text = captured[0]
    expected_id = doc_slice.slice_id  # "doc.pdf:0:24"
    assert tool_text.startswith("Passage:")
    assert f"[{expected_id}]" in tool_text
    assert "doc: doc.pdf" in tool_text
    assert "chars 0–24" in tool_text


# ---------------------------------------------------------------------------
# run() — cited-slice filtering integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_read_document_result_contains_only_cited_slices():
    """QueryResult.document_slices is filtered to slices cited by [slice_id] in the answer."""
    text = "A" * 500
    store = _make_document_store({"report.pdf": text})

    # Two sequential read_document calls; LLM cites only the first.
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "report.pdf", "offset": 0,   "length": 100}, call_id="c1"),
        _tool_result("read_document", {"doc_id": "report.pdf", "offset": 100, "length": 100}, call_id="c2"),
        _text_result("Based on [report.pdf:0:100], the answer is A."),
    )
    runner = _make_runner("myapp", llm, store)
    output = [c async for c in runner.run("read report")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    assert len(result.document_slices) == 1
    assert result.document_slices[0].slice_id == "report.pdf:0:100"


@pytest.mark.asyncio
async def test_run_read_document_fallback_all_slices_when_llm_cites_none():
    """QueryResult.document_slices falls back to all slices when the LLM uses no [slice_id] citations."""
    text = "B" * 500
    store = _make_document_store({"report.pdf": text})

    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "report.pdf", "offset": 0,   "length": 100}, call_id="c1"),
        _tool_result("read_document", {"doc_id": "report.pdf", "offset": 100, "length": 100}, call_id="c2"),
        _text_result("Based on the retrieved content, the answer is B."),
    )
    runner = _make_runner("myapp", llm, store)
    output = [c async for c in runner.run("read report")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    assert len(result.document_slices) == 2


# ---------------------------------------------------------------------------
# Bug fix: cross-type citation filtering
# When the LLM cites chunks, slices must not fall back to "all", and vice versa.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_citing_chunks_does_not_return_all_slices():
    """Main bug: answer cites a chunk ID → document_slices must be [], not all slices."""
    from cogbase.core.models import Chunk as ModelChunk

    text = "A" * 500
    doc_store = _make_document_store({"doc.pdf": text})
    store_chunks = [
        ModelChunk(chunk_id="doc_0_0", doc_id="doc.pdf", text="passage A", embedding=[0.1] * 4),
    ]

    llm = _make_llm(
        _tool_result("vector_search",  {"query": "q", "collection": "docs"},                  call_id="c1"),
        _tool_result("read_document",  {"doc_id": "doc.pdf", "offset": 0, "length": 100},     call_id="c2"),
        # LLM cites the chunk ID, not the slice ID
        _text_result("Based on [doc_0_0], the answer is passage A."),
    )
    runner = _make_runner(
        "myapp",
        llm,
        doc_store,
        vector_store=_fake_vector_store_with_chunks(store_chunks),
        embedder=_fake_embedder(),
    )
    output = [c async for c in runner.run("find passage")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "doc_0_0"
    # Slice was retrieved but not cited — must not fall back to all slices.
    assert result.document_slices == []


@pytest.mark.asyncio
async def test_run_citing_slices_does_not_return_all_chunks():
    """Symmetric bug: answer cites a slice ID → chunks must be [], not all chunks."""
    from cogbase.core.models import Chunk as ModelChunk

    text = "B" * 500
    doc_store = _make_document_store({"doc.pdf": text})
    store_chunks = [
        ModelChunk(chunk_id="doc_0_0", doc_id="doc.pdf", text="passage A", embedding=[0.1] * 4),
        ModelChunk(chunk_id="doc_0_1", doc_id="doc.pdf", text="passage B", embedding=[0.1] * 4),
    ]

    llm = _make_llm(
        _tool_result("vector_search",  {"query": "q", "collection": "docs"},              call_id="c1"),
        _tool_result("read_document",  {"doc_id": "doc.pdf", "offset": 0, "length": 100}, call_id="c2"),
        # LLM cites the slice ID (doc.pdf:0:100), not any chunk ID
        _text_result("Based on [doc.pdf:0:100], the answer is B."),
    )
    runner = _make_runner(
        "myapp",
        llm,
        doc_store,
        vector_store=_fake_vector_store_with_chunks(store_chunks),
        embedder=_fake_embedder(),
    )
    output = [c async for c in runner.run("find passage")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    assert len(result.document_slices) == 1
    assert result.document_slices[0].slice_id == "doc.pdf:0:100"
    # Chunks were retrieved but not cited — must not fall back to all chunks.
    assert result.chunks == []


# ---------------------------------------------------------------------------
# run() — input_tokens / output_tokens accumulation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_result_has_zero_tokens_when_no_usage_reported():
    """input_tokens and output_tokens default to 0 when the LLM reports no usage."""
    llm = _make_llm(_text_result("Hello."))
    runner = _make_runner("test", llm, _doc_store())
    chunks = [c async for c in runner.run("Hi")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_query_result_single_call_token_counts():
    """Tokens from a single LLM round are reflected in QueryResult."""
    llm = _make_llm_tracking(
        _result_with_usage(content="The answer.", input_tokens=100, output_tokens=25),
    )
    runner = _make_runner("test", llm, _doc_store())
    chunks = [c async for c in runner.run("Question?")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert result.input_tokens == 100
    assert result.output_tokens == 25


@pytest.mark.asyncio
async def test_query_result_accumulates_tokens_across_multiple_rounds():
    """Token counts are summed across a tool-call round and the final answer round."""
    tool_call_result = _result_with_usage(
        tool_calls=[{"id": "c1", "name": "shell", "arguments": '{"command": "echo hi"}'}],
        input_tokens=80,
        output_tokens=10,
    )
    answer_result = _result_with_usage(content="Done.", input_tokens=120, output_tokens=15)

    llm = _make_llm_tracking(tool_call_result, answer_result)
    runner = _make_runner("test", llm, _doc_store())
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="hi")):
        chunks = [c async for c in runner.run("run echo")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert result.input_tokens == 200   # 80 + 120
    assert result.output_tokens == 25   # 10 + 15


@pytest.mark.asyncio
async def test_query_result_passthrough_carries_accumulated_tokens():
    """Token counts are present in a passthrough QueryResult."""
    from cogbase.stores import CollectionSchema, FieldSchema, FieldType
    from cogbase.stores.structured.memory import InMemoryStructuredStore
    from pydantic import BaseModel as PydanticModel

    class BigRecord(PydanticModel):
        data: str

    store = InMemoryStructuredStore()
    schema = CollectionSchema(
        name="big",
        description="big collection",
        primary_fields=["data"],
        fields={"data": FieldSchema(type=FieldType.STRING)},
    )
    await store.create_collection(schema)
    await store.save("big", [BigRecord(data="x" * 25) for _ in range(400)])

    tool_call_result = _result_with_usage(
        tool_calls=[{"id": "c1", "name": "structured_lookup", "arguments": '{"collection": "big"}'}],
        input_tokens=50,
        output_tokens=8,
    )
    llm = _make_llm_tracking(tool_call_result)
    runner = _make_runner("test", llm, _doc_store(), structured_store=store, passthrough_token_threshold=2000)
    chunks = [c async for c in runner.run("dump big")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert result.passthrough is True
    assert result.input_tokens == 50
    assert result.output_tokens == 8


@pytest.mark.asyncio
async def test_query_result_max_calls_exceeded_carries_accumulated_tokens():
    """Token counts are present in the error QueryResult when max_calls is exceeded."""
    tool_result = _result_with_usage(
        tool_calls=[{"id": "c1", "name": "shell", "arguments": '{"command": "echo hi"}'}],
        input_tokens=60,
        output_tokens=5,
    )
    llm = _make_llm_tracking(tool_result, tool_result)
    runner = _make_runner("test", llm, _doc_store(), max_calls=2)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="ok")):
        chunks = [c async for c in runner.run("loop forever")]
    result = chunks[-1]
    assert isinstance(result, QueryResult)
    assert "unable to complete" in result.answer.lower()
    assert result.input_tokens == 120   # 60 × 2
    assert result.output_tokens == 10   # 5 × 2
