"""Unit tests for cogbase.core.runner.Runner.

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

from cogbase.core.runner import RunResult, Runner
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
    llm.complete = AsyncMock(side_effect=list(results))
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


def test_build_system_prompt_includes_runtime_context():
    skill = _make_skill("weather")
    runner = Runner(MagicMock())
    prompt = runner.build_system_prompt("base", skill, runtime_context={"user": "alice", "lang": "en"})
    assert "user: `alice`" in prompt
    assert "lang: `en`" in prompt


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
    assert isinstance(chunks[-1], RunResult)
    assert chunks[-1].answer == "It is sunny."


@pytest.mark.asyncio
async def test_run_skill_single_tool_call_then_answer():
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("weather"),                                 # select
        _tool_result("shell", {"command": "curl wttr.in/NYC"}), # tool call
        _text_result("weather"),                                 # re-select
        _text_result("The weather in NYC is 72°F."),            # answer
    )
    runner = Runner(llm, skills=skills)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="72°F, sunny")):
        chunks = [c async for c in runner.run("Weather in NYC?")]
    assert any("Executing: shell" in c for c in _str_chunks(chunks))
    assert _str_chunks(chunks)[-1] == "The weather in NYC is 72°F."


@pytest.mark.asyncio
async def test_run_skill_switches_between_iterations():
    skill_a = _make_skill("extract")
    skill_b = _make_skill("contradiction")
    llm = _make_llm(
        _text_result("extract"),                                  # select → extract
        _tool_result("shell", {"command": "python extract.py"}), # tool call
        _text_result("contradiction"),                            # re-select → contradiction
        _text_result("Found 2 contradictions."),                  # answer
    )
    runner = Runner(llm, skills=[skill_a, skill_b])
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="facts extracted")):
        chunks = [c async for c in runner.run("Find contradictions.")]
    status = [c for c in _str_chunks(chunks) if c.startswith("Using skill:")]
    assert "Using skill: extract..." in status
    assert "Using skill: contradiction..." in status
    assert _str_chunks(chunks)[-1] == "Found 2 contradictions."


@pytest.mark.asyncio
async def test_run_skill_unchanged_emits_status_once():
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("weather"),                             # select
        _tool_result("shell", {"command": "curl wttr.in"}), # tool call
        _text_result("weather"),                             # re-select (same)
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
    assert isinstance(chunks[-1], RunResult)


@pytest.mark.asyncio
async def test_run_max_calls_exceeded_yields_error():
    skills = [_make_skill("weather")]
    tool = _tool_result("shell", {"command": "echo hi"})
    llm = _make_llm(
        _text_result("weather"), tool, # round 1
        _text_result("weather"), tool, # round 2
    )
    runner = Runner(llm, max_calls=2, skills=skills)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="ok")):
        chunks = [c async for c in runner.run("Weather?")]
    assert any("unable to complete" in c.lower() for c in _str_chunks(chunks))
    assert isinstance(chunks[-1], RunResult)


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
    assert isinstance(chunks[-1], RunResult)
    assert chunks[-1].answer == "The answer is 42."


@pytest.mark.asyncio
async def test_run_retrieval_structured_lookup_populates_records():
    """structured_lookup results are accumulated in RunResult.structured_records."""
    from pydantic import BaseModel as PydanticModel
    from cogbase.stores.structured.memory import InMemoryStructuredStore
    from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType

    class Fact(PydanticModel):
        title: str

    store = InMemoryStructuredStore()
    schema = CollectionSchema(
        name="facts",
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
    assert isinstance(result, RunResult)
    assert len(result.structured_records) == 2
    assert result.passthrough is False
    assert result.answer == "Found: Foo, Bar."


@pytest.mark.asyncio
async def test_run_retrieval_passthrough_when_records_exceed_threshold():
    """structured_lookup with large result bypasses LLM synthesis."""
    from pydantic import BaseModel as PydanticModel
    from cogbase.stores.structured.memory import InMemoryStructuredStore
    from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType

    class BigRecord(PydanticModel):
        data: str

    store = InMemoryStructuredStore()
    schema = CollectionSchema(
        name="big",
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
    assert isinstance(result, RunResult)
    assert result.passthrough is True
    assert len(result.structured_records) == 400
    # LLM should NOT have been called for synthesis (only the one tool-call completion)
    assert llm.complete.call_count == 1


@pytest.mark.asyncio
async def test_run_retrieval_vector_search_populates_chunks():
    """vector_search results are accumulated in RunResult.chunks."""
    from cogbase.core.models import Chunk
    from cogbase.embeddings.base import EmbeddingBase
    from cogbase.stores.vector.base import VectorStoreBase

    class _FakeEmbedder(EmbeddingBase):
        async def embed(self, texts):
            return [[0.1] * 4 for _ in texts]

    class _FakeVectorStore(VectorStoreBase):
        async def upsert(self, collection, chunks): pass
        async def search(self, collection, embedding, top_k):
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
        default_vector_collection="docs",
    )
    chunks = [c async for c in runner.run("find relevant")]
    result = chunks[-1]
    assert isinstance(result, RunResult)
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
    from cogbase.stores.vector.base import VectorStoreBase

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
