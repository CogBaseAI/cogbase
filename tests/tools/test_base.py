"""Unit tests for cogbase.tools.base — Tool ABC and validation."""

import pytest

from cogbase.core.session import Session
from cogbase.tools.base import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Returns the input unchanged."

    async def run(self, input: dict, session: Session) -> dict:
        return input


# ---------------------------------------------------------------------------
# Valid subclass
# ---------------------------------------------------------------------------

def test_valid_subclass_instantiates():
    tool = EchoTool()
    assert tool.name == "echo"


# ---------------------------------------------------------------------------
# Missing required class variables
# ---------------------------------------------------------------------------

def test_missing_name_raises():
    with pytest.raises(TypeError, match="name"):
        class BadTool(Tool):
            description = "No name."

            async def run(self, input: dict, session: Session) -> dict:
                return {}


def test_missing_description_raises():
    with pytest.raises(TypeError, match="description"):
        class BadTool(Tool):
            name = "bad-tool"

            async def run(self, input: dict, session: Session) -> dict:
                return {}


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

def test_name_uppercase_raises():
    with pytest.raises(TypeError, match="invalid"):
        class BadTool(Tool):
            name = "BadTool"
            description = "Bad name."

            async def run(self, input: dict, session: Session) -> dict:
                return {}


def test_name_leading_hyphen_raises():
    with pytest.raises(TypeError, match="invalid"):
        class BadTool(Tool):
            name = "-bad"
            description = "Bad name."

            async def run(self, input: dict, session: Session) -> dict:
                return {}


def test_name_consecutive_hyphens_raises():
    with pytest.raises(TypeError, match="invalid"):
        class BadTool(Tool):
            name = "bad--tool"
            description = "Bad name."

            async def run(self, input: dict, session: Session) -> dict:
                return {}


def test_name_too_long_raises():
    with pytest.raises(TypeError, match="64"):
        class BadTool(Tool):
            name = "a" * 65
            description = "Name too long."

            async def run(self, input: dict, session: Session) -> dict:
                return {}


def test_name_empty_raises():
    with pytest.raises(TypeError, match="empty"):
        class BadTool(Tool):
            name = ""
            description = "Empty name."

            async def run(self, input: dict, session: Session) -> dict:
                return {}


# ---------------------------------------------------------------------------
# Description validation
# ---------------------------------------------------------------------------

def test_description_too_long_raises():
    with pytest.raises(TypeError, match="1024"):
        class BadTool(Tool):
            name = "long-desc"
            description = "x" * 1025

            async def run(self, input: dict, session: Session) -> dict:
                return {}


def test_description_empty_raises():
    with pytest.raises(TypeError, match="empty"):
        class BadTool(Tool):
            name = "empty-desc"
            description = ""

            async def run(self, input: dict, session: Session) -> dict:
                return {}


# ---------------------------------------------------------------------------
# Abstract enforcement
# ---------------------------------------------------------------------------

def test_abstract_run_not_implemented():
    with pytest.raises(TypeError):
        class NoRun(Tool):
            name = "no-run"
            description = "Missing run."
        NoRun()
