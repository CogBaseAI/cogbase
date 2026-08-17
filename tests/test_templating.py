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


# --- strings stay strings ------------------------------------------------------


@pytest.mark.parametrize("value", ["211", "3", "007", "F", "0.1.0", "2026-01-01"])
def test_a_lone_expression_returns_a_string_unchanged(value: str) -> None:
    """Jinja's ``native_concat`` runs ``literal_eval`` over a single node even when
    that node is already a ``str``, so ``"211"`` came back as the integer ``211``
    while ``"007"`` came back as a string — the same column retyped or not depending
    on its contents. Every caller renders these into equality filters and record
    fields, where the wrong type does not raise, it just stops matching.
    """
    rendered = render_value("{{ record.part }}", {"record": {"part": value}})
    assert rendered == value
    assert isinstance(rendered, str)


def test_a_lone_expression_still_returns_non_string_objects_natively() -> None:
    """The reason the native environment is used at all: ``foreach`` and ``records``
    templates resolve to real lists and dicts, not to their reprs."""
    ctx = {"steps": {"load": {"records": [{"id": "a"}, {"id": "b"}]}}}
    assert render_value("{{ steps.load.records }}", ctx) == [{"id": "a"}, {"id": "b"}]
    assert render_value("{{ steps.load.records[0] }}", ctx) == {"id": "a"}
    assert render_value("{{ n }}", {"n": 12}) == 12
    assert render_value("{{ ok }}", {"ok": True}) is True
    assert render_value("{{ nothing }}", {"nothing": None}) is None


def test_multi_node_templates_keep_jinja_behaviour() -> None:
    """Unchanged: text genuinely being assembled is assembled as text."""
    ctx = {"record": {"part": "211", "subpart": "F"}}
    assert render_value("{{ record.subpart }}\n{{ record.part }}", ctx) == "F\n211"
    assert render_value("part {{ record.part }}", ctx) == "part 211"


def test_an_empty_template_renders_to_none() -> None:
    assert render_value("", {}) is None
