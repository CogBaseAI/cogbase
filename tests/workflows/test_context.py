"""Unit tests for cogbase.workflows.context — Jinja2 template rendering."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from cogbase.workflows.context import render_value


class _Point(BaseModel):
    x: float
    y: float


# ---------------------------------------------------------------------------
# Scalars and string rendering
# ---------------------------------------------------------------------------

class TestRenderValueStrings:
    def test_plain_string_unchanged(self):
        assert render_value("hello", {}) == "hello"

    def test_single_expression_returns_string(self):
        ctx = {"doc_id": "abc"}
        assert render_value("{{ doc_id }}", ctx) == "abc"

    def test_mixed_text_returns_string(self):
        ctx = {"name": "world"}
        assert render_value("hello {{ name }}!", ctx) == "hello world!"

    def test_multiline_mixed_returns_string(self):
        ctx = {"a": "foo", "b": "bar"}
        result = render_value("{{ a }}\n{{ b }}", ctx)
        assert result == "foo\nbar"

    def test_integer_expression_returns_int(self):
        ctx = {"n": 42}
        result = render_value("{{ n }}", ctx)
        assert result == 42
        assert isinstance(result, int)

    def test_float_expression_returns_float(self):
        ctx = {"v": 3.14}
        result = render_value("{{ v }}", ctx)
        assert result == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Native Python types returned from single expressions
# ---------------------------------------------------------------------------

class TestRenderValueNativeTypes:
    def test_list_expression_returns_list(self):
        ctx = {"items": [1, 2, 3]}
        result = render_value("{{ items }}", ctx)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_dict_expression_returns_dict(self):
        ctx = {"record": {"clause_id": "c1", "text": "foo"}}
        result = render_value("{{ record }}", ctx)
        assert result == {"clause_id": "c1", "text": "foo"}
        assert isinstance(result, dict)

    def test_pydantic_model_expression_returns_model(self):
        model = _Point(x=1.0, y=2.0)
        ctx = {"point": model}
        result = render_value("{{ point }}", ctx)
        assert result is model

    def test_nested_dict_access(self):
        ctx = {"steps": {"load": {"records": [{"id": "r1"}]}}}
        result = render_value("{{ steps.load.records }}", ctx)
        assert result == [{"id": "r1"}]

    def test_none_expression_returns_none(self):
        ctx = {"val": None}
        result = render_value("{{ val }}", ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Attribute access on Pydantic models
# ---------------------------------------------------------------------------

class TestRenderValueModelAccess:
    def test_model_attribute_via_dot(self):
        model = _Point(x=3.0, y=4.0)
        ctx = {"item": model}
        assert render_value("{{ item.x }}", ctx) == 3.0

    def test_model_field_in_mixed_string(self):
        model = _Point(x=1.5, y=2.5)
        ctx = {"item": model}
        result = render_value("x={{ item.x }}, y={{ item.y }}", ctx)
        assert result == "x=1.5, y=2.5"


# ---------------------------------------------------------------------------
# Non-string input values — passed through unchanged
# ---------------------------------------------------------------------------

class TestRenderValuePassthrough:
    def test_integer_passthrough(self):
        assert render_value(42, {}) == 42

    def test_none_passthrough(self):
        assert render_value(None, {}) is None

    def test_bool_passthrough(self):
        assert render_value(True, {}) is True


# ---------------------------------------------------------------------------
# Recursive rendering of dicts and lists
# ---------------------------------------------------------------------------

class TestRenderValueRecursive:
    def test_dict_values_rendered(self):
        ctx = {"doc_id": "abc"}
        result = render_value({"field": "{{ doc_id }}"}, ctx)
        assert result == {"field": "abc"}

    def test_list_elements_rendered(self):
        ctx = {"doc_id": "abc"}
        result = render_value(["{{ doc_id }}", "literal"], ctx)
        assert result == ["abc", "literal"]

    def test_nested_dict_rendered(self):
        ctx = {"x": 10, "y": 20}
        result = render_value({"a": {"b": "{{ x }}", "c": "{{ y }}"}}, ctx)
        assert result == {"a": {"b": 10, "c": 20}}

    def test_list_of_dicts_rendered(self):
        ctx = {"v": "hello"}
        result = render_value([{"key": "{{ v }}"}], ctx)
        assert result == [{"key": "hello"}]

    def test_non_string_values_in_dict_passed_through(self):
        result = render_value({"n": 5, "flag": True}, {})
        assert result == {"n": 5, "flag": True}


# ---------------------------------------------------------------------------
# Error on undefined variable
# ---------------------------------------------------------------------------

class TestRenderValueErrors:
    def test_undefined_attribute_on_undefined_raises(self):
        # Attribute access on an undefined variable triggers StrictUndefined.
        from jinja2 import UndefinedError
        ctx = {"steps": {}}
        with pytest.raises(UndefinedError):
            render_value("{{ steps.missing.records }}", ctx)

    def test_undefined_in_mixed_string_raises(self):
        # Coercing an undefined to string (mixed template) also raises.
        from jinja2 import UndefinedError
        with pytest.raises(UndefinedError):
            render_value("prefix-{{ no_such_var }}-suffix", {})
