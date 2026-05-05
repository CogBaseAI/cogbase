"""Jinja2-based template rendering for workflow step parameters.

Uses ``NativeEnvironment`` so that a pure ``{{ expr }}`` template returns the
native Python value (list, dict, BaseModel …) rather than a string.  Templates
that mix literal text with expressions are always rendered as strings.
"""

from __future__ import annotations

from typing import Any

try:
    from jinja2 import StrictUndefined
    from jinja2.nativetypes import NativeEnvironment

    _env = NativeEnvironment(undefined=StrictUndefined)
except ImportError:  # pragma: no cover
    _env = None  # type: ignore[assignment]


def render_value(value: Any, ctx: dict) -> Any:
    """Render *value* recursively against *ctx*.

    - ``str`` values are treated as Jinja2 templates.
    - ``list`` / ``dict`` values are recursed into element-by-element.
    - All other values are returned unchanged.
    """
    if _env is None:
        raise RuntimeError(
            "jinja2 is required for workflow template rendering: "
            "pip install 'cogbase[api]'"
        )
    if isinstance(value, str):
        return _env.from_string(value).render(**ctx)
    if isinstance(value, list):
        return [render_value(item, ctx) for item in value]
    if isinstance(value, dict):
        return {k: render_value(v, ctx) for k, v in value.items()}
    return value
