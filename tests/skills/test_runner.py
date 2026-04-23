"""Tests for SkillRunner: select, build_system_prompt, run loop, execute_tool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.llms.base import ChatMessage, CompletionResult
from cogbase.skills.runner import SkillRunner
from cogbase.skills.skill import Skill


# ---------------------------------------------------------------------------
# Helpers
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
    """Return a mock LLMBase whose complete() returns *results* in sequence."""
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=list(results))
    return llm


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_returns_matching_skill():
    skills = [_make_skill("weather"), _make_skill("model-usage")]
    llm = _make_llm(_text_result("weather"))
    runner = SkillRunner(llm)
    result = await runner.select(skills, "What's the weather?")
    assert result is skills[0]


@pytest.mark.asyncio
async def test_select_returns_none_for_no_match():
    skills = [_make_skill("weather")]
    llm = _make_llm(_text_result("none"))
    runner = SkillRunner(llm)
    result = await runner.select(skills, "Tell me a joke")
    assert result is None


@pytest.mark.asyncio
async def test_select_returns_none_for_unknown_skill_name():
    skills = [_make_skill("weather")]
    llm = _make_llm(_text_result("nonexistent"))
    runner = SkillRunner(llm)
    result = await runner.select(skills, "something")
    assert result is None


@pytest.mark.asyncio
async def test_select_empty_skills_returns_none_without_llm_call():
    llm = MagicMock()
    llm.complete = AsyncMock()
    runner = SkillRunner(llm)
    result = await runner.select([], "anything")
    assert result is None
    llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# build_system_prompt()
# ---------------------------------------------------------------------------

def test_build_system_prompt_includes_skill_markdown():
    skill = _make_skill("weather", markdown="# Weather\nRun curl.")
    runner = SkillRunner(MagicMock())
    prompt = runner.build_system_prompt("You are helpful.", skill)
    assert "# Weather\nRun curl." in prompt
    assert "Active Skill: weather" in prompt


def test_build_system_prompt_includes_runtime_context():
    skill = _make_skill("weather")
    runner = SkillRunner(MagicMock())
    prompt = runner.build_system_prompt("base", skill, runtime_context={"user": "alice", "lang": "en"})
    assert "user: `alice`" in prompt
    assert "lang: `en`" in prompt


def test_build_system_prompt_includes_metadata():
    skill = _make_skill("weather")
    skill.metadata = {"requires": {"bins": ["curl"]}}
    runner = SkillRunner(MagicMock())
    prompt = runner.build_system_prompt("base", skill)
    assert "curl" in prompt


# ---------------------------------------------------------------------------
# run() — happy paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_no_tools_yields_final_answer():
    """LLM answers immediately with no tool calls."""
    skills = [_make_skill("weather")]
    # select returns "weather", complete returns text
    llm = _make_llm(
        _text_result("weather"),       # select call
        _text_result("It is sunny."),  # run call
    )
    runner = SkillRunner(llm)
    chunks = [c async for c in runner.run(skills, "Weather?")]
    assert any("Using skill: weather" in c for c in chunks)
    assert chunks[-1] == "It is sunny."


@pytest.mark.asyncio
async def test_run_single_tool_call_then_answer():
    """LLM calls shell once, then gives a final text answer."""
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("weather"),                                  # select
        _tool_result("shell", {"command": "curl wttr.in/NYC"}),  # tool call
        _text_result("weather"),                                  # re-select after tool
        _text_result("The weather in NYC is 72°F."),             # final answer
    )
    runner = SkillRunner(llm)

    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="72°F, sunny")):
        chunks = [c async for c in runner.run(skills, "Weather in NYC?")]

    assert any("Executing: shell" in c for c in chunks)
    assert chunks[-1] == "The weather in NYC is 72°F."


@pytest.mark.asyncio
async def test_run_switches_skill_between_iterations():
    """Agent switches from skill-a to skill-b after first tool call."""
    skill_a = _make_skill("extract")
    skill_b = _make_skill("contradiction")
    skills = [skill_a, skill_b]

    llm = _make_llm(
        _text_result("extract"),                                        # select → extract
        _tool_result("shell", {"command": "python extract.py"}),       # tool call
        _text_result("contradiction"),                                  # re-select → contradiction
        _text_result("Found 2 contradictions."),                        # final answer
    )
    runner = SkillRunner(llm)

    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="facts extracted")):
        chunks = [c async for c in runner.run(skills, "Find contradictions in doc.")]

    skill_status = [c for c in chunks if c.startswith("Using skill:")]
    assert "Using skill: extract..." in skill_status
    assert "Using skill: contradiction..." in skill_status
    assert chunks[-1] == "Found 2 contradictions."


@pytest.mark.asyncio
async def test_run_no_skill_selected_answers_directly():
    """When no skill applies the LLM answers without skill context."""
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("none"),           # select → no skill
        _text_result("I don't know."),  # direct answer
    )
    runner = SkillRunner(llm)
    chunks = [c async for c in runner.run(skills, "What is 2+2?")]
    assert chunks[-1] == "I don't know."
    assert not any("Using skill" in c for c in chunks)


# ---------------------------------------------------------------------------
# run() — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_max_calls_exceeded_yields_error():
    """Runner gives up after max_calls tool iterations."""
    skills = [_make_skill("weather")]
    # select and run both go through llm.complete — results must be interleaved:
    # iteration 1: select→"weather", run→tool_call
    # iteration 2: select→"weather", run→tool_call
    # loop exits (call_count == max_calls=2), yields error message
    tool = _tool_result("shell", {"command": "echo hi"})
    llm = _make_llm(
        _text_result("weather"), tool,   # iteration 1
        _text_result("weather"), tool,   # iteration 2
    )
    runner = SkillRunner(llm, max_calls=2)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="ok")):
        chunks = [c async for c in runner.run(skills, "Weather?")]
    assert any("unable to complete" in c.lower() for c in chunks)


@pytest.mark.asyncio
async def test_run_skill_unchanged_does_not_repeat_status():
    """If the same skill is re-selected, 'Using skill:' is only yielded once."""
    skills = [_make_skill("weather")]
    llm = _make_llm(
        _text_result("weather"),                                  # select
        _tool_result("shell", {"command": "curl wttr.in"}),      # tool call
        _text_result("weather"),                                  # re-select (same)
        _text_result("Sunny."),                                   # final answer
    )
    runner = SkillRunner(llm)
    with patch.object(runner, "_execute_tool", new=AsyncMock(return_value="sunny")):
        chunks = [c async for c in runner.run(skills, "Weather?")]
    status_chunks = [c for c in chunks if "Using skill" in c]
    assert len(status_chunks) == 1


# ---------------------------------------------------------------------------
# _execute_tool()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_tool_python_returns_stdout():
    runner = SkillRunner(MagicMock())
    output = await runner._execute_tool("python", {"code": "print('hello')"})
    assert output == "hello"


@pytest.mark.asyncio
async def test_execute_tool_shell_returns_stdout():
    runner = SkillRunner(MagicMock())
    output = await runner._execute_tool("shell", {"command": "echo hi"})
    assert output == "hi"


@pytest.mark.asyncio
async def test_execute_tool_unknown_returns_error():
    runner = SkillRunner(MagicMock())
    output = await runner._execute_tool("nonexistent", {})
    assert "Unknown tool" in output


@pytest.mark.asyncio
async def test_execute_tool_python_bad_code_returns_error():
    runner = SkillRunner(MagicMock())
    output = await runner._execute_tool("python", {"code": "raise ValueError('boom')"})
    assert output  # stderr captured, not empty


# ---------------------------------------------------------------------------
# compact_messages()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_messages_returns_two_messages():
    llm = _make_llm(_text_result("Summary: did X then Y."))
    runner = SkillRunner(llm)
    history: list[ChatMessage] = [
        {"role": "user", "content": "step 1"},
        {"role": "assistant", "content": "done step 1"},
    ]
    compacted = await runner.compact_messages("You are helpful.", history)
    assert len(compacted) == 2
    assert compacted[0]["role"] == "system"
    assert "Summary" in compacted[1]["content"]
