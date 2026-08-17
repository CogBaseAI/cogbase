"""Tests for the shared template renderer.

``render_value`` itself is covered by ``tests/workflows/test_context.py``, which
still imports it from its original home — that re-export is part of the contract.
What is tested here is ``render_defined``, and the jinja2 behaviour it exists to
correct.
"""

from __future__ import annotations

import pytest

from cogbase.templating import jinja_available, render_defined, render_value
from cogbase.workflows.context import render_value as reexported


def test_workflow_import_path_still_resolves_to_the_same_function() -> None:
    assert reexported is render_value


def test_jinja_is_available_in_this_install() -> None:
    assert jinja_available()


def test_render_defined_returns_the_native_value() -> None:
    assert render_defined("{{ doc.n }}", {"doc": {"n": 12}}, what="fields.n") == 12


def test_render_defined_renders_mixed_text_as_a_string() -> None:
    result = render_defined("v{{ doc.n }}", {"doc": {"n": 12}}, what="fields.v")
    assert result == "v12"


def test_render_defined_raises_on_a_missing_name() -> None:
    """The reason this function exists. ``NativeEnvironment`` returns a lone
    ``{{ expr }}`` node raw, so ``StrictUndefined`` is never *used* and never
    raises — the Undefined object itself comes back and would be written into a
    record as a value."""
    with pytest.raises(ValueError, match="fields.subpart"):
        render_defined("{{ doc.metadata.subpart }}", {"doc": {"metadata": {}}}, what="fields.subpart")


def test_render_value_alone_does_not_raise_on_a_missing_name() -> None:
    """Documents the lenient behaviour ``render_defined`` wraps. The workflow
    ``structured-save`` overlay has rendered this way since it shipped; changing it
    is its own change, not a side effect of adding the pipeline overlay."""
    from jinja2 import is_undefined

    assert is_undefined(render_value("{{ doc.metadata.subpart }}", {"doc": {"metadata": {}}}))


def test_render_defined_accepts_an_explicit_default() -> None:
    result = render_defined(
        "{{ doc.metadata.get('subpart') | default(None, true) }}",
        {"doc": {"metadata": {}}},
        what="fields.subpart",
    )
    assert result is None
