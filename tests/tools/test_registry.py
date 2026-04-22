"""Unit tests for cogbase.tools.registry — ToolRegistry."""

import pytest

from cogbase.core.session import Session
from cogbase.tools.base import Tool
from cogbase.tools.registry import ToolInfo, ToolRegistry


class AlphaTool(Tool):
    name = "alpha"
    description = "Alpha tool."

    async def run(self, input: dict, session: Session) -> dict:
        return {}


class BetaTool(Tool):
    name = "beta"
    description = "Beta tool."

    async def run(self, input: dict, session: Session) -> dict:
        return {}


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------

def test_register_and_get():
    reg = ToolRegistry()
    tool = AlphaTool()
    reg.register(tool)
    assert reg.get("alpha") is tool


def test_duplicate_registration_raises():
    reg = ToolRegistry()
    reg.register(AlphaTool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(AlphaTool())


def test_get_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="unknown"):
        reg.get("unknown")


def test_get_error_lists_known_tools():
    reg = ToolRegistry()
    reg.register(AlphaTool())
    with pytest.raises(KeyError, match="alpha"):
        reg.get("missing")


def test_get_error_shows_none_when_empty():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match=r"\(none\)"):
        reg.get("anything")


# ---------------------------------------------------------------------------
# deregister
# ---------------------------------------------------------------------------

def test_deregister_removes_tool():
    reg = ToolRegistry()
    reg.register(AlphaTool())
    reg.deregister("alpha")
    with pytest.raises(KeyError):
        reg.get("alpha")


def test_deregister_noop_for_unknown():
    reg = ToolRegistry()
    reg.deregister("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# list_tools / list_builtin_tools
# ---------------------------------------------------------------------------

def test_list_tools_empty():
    assert ToolRegistry().list_tools() == []


def test_list_tools_returns_all_sorted():
    reg = ToolRegistry()
    reg.register(BetaTool())
    reg.register(AlphaTool())
    infos = reg.list_tools()
    assert [ti.name for ti in infos] == ["alpha", "beta"]


def test_list_tools_builtin_flag():
    reg = ToolRegistry()
    reg.register(AlphaTool(), builtin=True)
    reg.register(BetaTool(), builtin=False)
    infos = {ti.name: ti for ti in reg.list_tools()}
    assert infos["alpha"].builtin is True
    assert infos["beta"].builtin is False


def test_list_builtin_tools_filters():
    reg = ToolRegistry()
    reg.register(AlphaTool(), builtin=True)
    reg.register(BetaTool(), builtin=False)
    builtins = reg.list_builtin_tools()
    assert len(builtins) == 1
    assert builtins[0].name == "alpha"


def test_list_builtin_tools_empty_registry():
    assert ToolRegistry().list_builtin_tools() == []


def test_deregister_also_removes_builtin_flag():
    reg = ToolRegistry()
    reg.register(AlphaTool(), builtin=True)
    reg.deregister("alpha")
    assert reg.list_builtin_tools() == []


def test_tool_info_fields():
    reg = ToolRegistry()
    reg.register(AlphaTool(), builtin=True)
    info = reg.list_tools()[0]
    assert isinstance(info, ToolInfo)
    assert info.name == "alpha"
    assert info.description == "Alpha tool."
    assert info.builtin is True
