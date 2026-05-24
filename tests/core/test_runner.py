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

from cogbase.core.query_runner import DocumentSlice, QueryResult, QueryRunner as Runner
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
    return llm


def _str_chunks(chunks: list) -> list[str]:
    return [c for c in chunks if isinstance(c, str)]


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_returns_matching_skill():
    skills = [_make_skill("weather"), _make_skill("model-usage")]
    llm = _make_llm(_text_result("weather"))
    runner = Runner(llm, skills=skills)
    result = await runner.select("What's the weather?")
    assert result is skills[0]


@pytest.mark.asyncio
async def test_select_returns_none_for_no_match():
    skills = [_make_skill("weather")]
    llm = _make_llm(_text_result("none"))
    runner = Runner(llm, skills=skills)
    result = await runner.select("Tell me a joke")
    assert result is None


@pytest.mark.asyncio
async def test_select_returns_none_for_unknown_skill_name():
    skills = [_make_skill("weather")]
    llm = _make_llm(_text_result("nonexistent"))
    runner = Runner(llm, skills=skills)
    result = await runner.select("something")
    assert result is None


@pytest.mark.asyncio
async def test_select_empty_skills_returns_none_without_llm_call():
    llm = MagicMock()
    llm.complete = AsyncMock()
    runner = Runner(llm)
    result = await runner.select("anything")
    assert result is None
    llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# build_system_prompt()
# ---------------------------------------------------------------------------

def test_build_system_prompt_includes_skill_markdown():
    skill = _make_skill("weather", markdown="# Weather\nRun curl.")
    runner = Runner(MagicMock())
    prompt = runner.build_system_prompt("You are helpful.", skill)
    assert "# Weather\nRun curl." in prompt
    assert "Active Skill: weather" in prompt



def test_build_system_prompt_includes_metadata():
    skill = _make_skill("weather")
    skill.metadata = {"requires": {"bins": ["curl"]}}
    runner = Runner(MagicMock())
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
    runner = Runner(llm, skills=skills)
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
    runner = Runner(llm, skills=skills)
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
    runner = Runner(llm, skills=skills)
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
    runner = Runner(llm, skills=skills)
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
    runner = Runner(llm, max_calls=2, skills=skills)
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
    runner = Runner(llm)
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
    runner = Runner(llm, structured_store=store)
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
    runner = Runner(llm, structured_store=store, passthrough_token_threshold=2000)
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
    runner = Runner(
        llm,
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
    from cogbase.stores.structured.memory import InMemoryStructuredStore
    runner = Runner(MagicMock(), structured_store=InMemoryStructuredStore())
    names = [t["name"] for t in runner._tool_defs]
    assert "structured_lookup" in names
    assert "vector_search" not in names


def test_tool_defs_vector_only():
    from cogbase.embeddings.base import EmbeddingBase
    from cogbase.stores import VectorStoreBase

    class _V(VectorStoreBase):
        async def upsert(self, c, chunks): pass
        async def search(self, c, e, k): return []
        async def delete(self, c, d): pass
        async def delete_collection(self, c): pass
        async def create_collection(self, s): pass

    class _E(EmbeddingBase):
        async def embed(self, texts): return [[0.0]]

    runner = Runner(MagicMock(), vector_store=_V(), embedder=_E())
    names = [t["name"] for t in runner._tool_defs]
    assert "vector_search" in names
    assert "structured_lookup" not in names


def test_tool_defs_no_stores_empty():
    runner = Runner(MagicMock())
    assert runner._tool_defs == []


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
    runner = Runner(llm, system_tools=[system_tool])
    chunks = [c async for c in runner.run("run my tool")]
    assert called_with == [{"arg": "hello"}]
    assert _str_chunks(chunks)[-1] == "Done."


# ---------------------------------------------------------------------------
# _execute_tool()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_tool_python_returns_stdout():
    runner = Runner(MagicMock())
    output = await runner._execute_tool("python", {"code": "print('hello')"})
    assert output == "hello"


@pytest.mark.asyncio
async def test_execute_tool_shell_returns_stdout():
    runner = Runner(MagicMock())
    output = await runner._execute_tool("shell", {"command": "echo hi"})
    assert output == "hi"


@pytest.mark.asyncio
async def test_execute_tool_unknown_returns_error():
    runner = Runner(MagicMock())
    output = await runner._execute_tool("nonexistent", {})
    assert "Unknown tool" in output


@pytest.mark.asyncio
async def test_execute_tool_python_bad_code_returns_stderr():
    runner = Runner(MagicMock())
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
    runner = Runner(MagicMock(), system_tools=[system_tool])
    output = await runner._execute_tool("boom", {})
    assert "Tool error" in output
    assert "exploded" in output


# ---------------------------------------------------------------------------
# run() — read_document tool
# ---------------------------------------------------------------------------

def _make_document_store(docs: dict[str, str]) -> MagicMock:
    store = MagicMock()
    async def _load(collection, doc_id):
        if doc_id not in docs:
            raise KeyError(doc_id)
        return docs[doc_id]
    store.load = AsyncMock(side_effect=_load)
    return store


def test_tool_defs_read_document_when_document_store_and_app_name_set():
    store = _make_document_store({})
    runner = Runner(MagicMock(), document_store=store, app_name="myapp")
    names = [t["name"] for t in runner._tool_defs]
    assert "read_document" in names


def test_tool_defs_read_document_not_added_without_app_name():
    store = _make_document_store({})
    runner = Runner(MagicMock(), document_store=store, app_name=None)
    names = [t["name"] for t in runner._tool_defs]
    assert "read_document" not in names


def test_tool_defs_read_document_not_added_without_document_store():
    runner = Runner(MagicMock(), app_name="myapp")
    names = [t["name"] for t in runner._tool_defs]
    assert "read_document" not in names


@pytest.mark.asyncio
async def test_run_read_document_returns_slice():
    text = "A" * 100 + "B" * 100 + "C" * 100
    store = _make_document_store({"doc-1": text})
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "doc-1", "offset": 50, "length": 100}),
        _text_result("Got the slice."),
    )
    runner = Runner(llm, document_store=store, app_name="myapp")
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
    runner = Runner(llm, document_store=store, app_name="myapp")
    with patch.object(Runner, "_run_read_document", _capturing):
        [c async for c in runner.run("read doc")]

    assert len(captured_tool_outputs) == 1
    doc_slice, tool_text = captured_tool_outputs[0]
    assert isinstance(doc_slice, DocumentSlice)
    assert "hello hello " in tool_text
    assert "chars 0–12 of 300" in tool_text


@pytest.mark.asyncio
async def test_run_read_document_missing_doc_returns_error():
    store = _make_document_store({})
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "missing"}),
        _text_result("Could not find it."),
    )
    runner = Runner(llm, document_store=store, app_name="myapp")
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
    runner = Runner(llm, document_store=store, app_name="myapp")
    with patch.object(Runner, "_run_read_document", _cap):
        [c async for c in runner.run("read big")]

    assert len(captured) == 1
    doc_slice, tool_text = captured[0]
    assert isinstance(doc_slice, DocumentSlice)
    assert "chars 0–10000 of 20000" in tool_text


@pytest.mark.asyncio
async def test_run_read_document_unavailable_without_store():
    """read_document called without a configured store returns (None, error_string)."""
    runner = Runner(MagicMock())
    doc_slice, output = await runner._run_read_document({"doc_id": "doc-1"})
    assert doc_slice is None
    assert "unavailable" in output


@pytest.mark.asyncio
async def test_run_read_document_populates_document_slices():
    """Successful read_document calls are accumulated in QueryResult.document_slices."""
    text = "X" * 500
    store = _make_document_store({"doc-1": text})
    llm = _make_llm(
        _tool_result("read_document", {"doc_id": "doc-1", "offset": 10, "length": 50}),
        _text_result("Done."),
    )
    runner = Runner(llm, document_store=store, app_name="myapp")
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
    runner = Runner(llm, document_store=store, app_name="myapp")
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
    runner = Runner(llm)
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
    assert "id=contract_001_0" in output
    assert "doc: contract_001" in output
    assert "Payment terms are net 30." in output


def test_format_chunks_includes_char_offsets_when_present():
    from cogbase.core.query_runner import _format_chunks
    from cogbase.core.models import Chunk

    chunk = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="text", char_offset=100, char_length=50)
    output = _format_chunks([chunk])
    assert "char_offset: 100" in output
    assert "char_length: 50" in output


def test_format_chunks_empty_returns_placeholder():
    from cogbase.core.query_runner import _format_chunks

    assert _format_chunks([]) == "(no passages found)"


# ---------------------------------------------------------------------------
# _filter_cited_chunks()
# ---------------------------------------------------------------------------

def test_filter_cited_chunks_returns_cited_subset():
    from cogbase.core.query_runner import _filter_cited_chunks
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="alpha")
    c2 = Chunk(chunk_id="doc_0_1", doc_id="doc_0", text="beta")
    c3 = Chunk(chunk_id="doc_1_0", doc_id="doc_1", text="gamma")

    result = _filter_cited_chunks(
        "Based on [doc_0_0] and [doc_1_0], the answer is clear.",
        [c1, c2, c3],
    )
    assert result == [c1, c3]


def test_filter_cited_chunks_fallback_when_no_citations():
    from cogbase.core.query_runner import _filter_cited_chunks
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="alpha")
    c2 = Chunk(chunk_id="doc_0_1", doc_id="doc_0", text="beta")

    result = _filter_cited_chunks("Based on the documents, here is the answer.", [c1, c2])
    assert result == [c1, c2]


def test_filter_cited_chunks_ignores_unknown_bracket_patterns():
    from cogbase.core.query_runner import _filter_cited_chunks
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="alpha")

    # Numbered refs like [1], [2] are not valid chunk_ids — should trigger fallback.
    result = _filter_cited_chunks("See [1] for details, also [2] is relevant.", [c1])
    assert result == [c1]


def test_filter_cited_chunks_empty_input_returns_empty():
    from cogbase.core.query_runner import _filter_cited_chunks

    assert _filter_cited_chunks("any answer with [doc_0_0]", []) == []


def test_filter_cited_chunks_preserves_all_chunks_order():
    from cogbase.core.query_runner import _filter_cited_chunks
    from cogbase.core.models import Chunk

    c1 = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="first")
    c2 = Chunk(chunk_id="doc_0_1", doc_id="doc_0", text="second")
    c3 = Chunk(chunk_id="doc_0_2", doc_id="doc_0", text="third")

    # LLM cited in reverse order — result should follow all_chunks order.
    result = _filter_cited_chunks("See [doc_0_2] and [doc_0_0].", [c1, c2, c3])
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
    runner = Runner(
        llm,
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
    runner = Runner(
        llm,
        vector_store=_fake_vector_store_with_chunks(store_chunks),
        embedder=_fake_embedder(),
    )
    output = [c async for c in runner.run("find passage")]
    result = output[-1]
    assert isinstance(result, QueryResult)
    assert len(result.chunks) == 2
