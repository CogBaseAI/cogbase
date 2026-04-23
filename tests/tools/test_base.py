"""Unit tests for tool name/description validation enforced by ToolRegistry.register."""

import pytest

from cogbase.llms.base import SystemTool, ToolDefinition
from cogbase.tools.registry import ToolRegistry


def _tool(name: str, description: str = "A tool.") -> SystemTool:
    return SystemTool(
        definition=ToolDefinition(name=name, description=description, parameters={}),
        handler=lambda inputs: "{}",
    )


def _register(name: str, description: str = "A tool.") -> None:
    ToolRegistry().register(_tool(name, description))


# ---------------------------------------------------------------------------
# Valid tool
# ---------------------------------------------------------------------------

def test_valid_tool_registers():
    reg = ToolRegistry()
    tool = _tool("echo")
    reg.register(tool)
    assert reg.get("echo") is tool


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

def test_name_uppercase_raises():
    with pytest.raises(ValueError, match="invalid"):
        _register("BadTool")


def test_name_leading_hyphen_raises():
    with pytest.raises(ValueError, match="invalid"):
        _register("-bad")


def test_name_consecutive_hyphens_raises():
    with pytest.raises(ValueError, match="invalid"):
        _register("bad--tool")


def test_name_too_long_raises():
    with pytest.raises(ValueError, match="64"):
        _register("a" * 65)


def test_name_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        _register("")


# ---------------------------------------------------------------------------
# Description validation
# ---------------------------------------------------------------------------

def test_description_too_long_raises():
    with pytest.raises(ValueError, match="1024"):
        _register("long-desc", description="x" * 1025)


def test_description_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        _register("empty-desc", description="")
