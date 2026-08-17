"""Jinja2-based template rendering, shared by the workflow and pipeline layers.

Uses ``NativeEnvironment`` so that a pure ``{{ expr }}`` template returns the
native Python value (list, dict, BaseModel …) rather than a string.  Templates
that mix literal text with expressions are always rendered as strings.

One correction to that environment, in ``_concat`` below: a lone ``{{ expr }}``
returns the context object *unchanged*, including when it is a ``str``.  Jinja's
own ``native_concat`` runs ``ast.literal_eval`` over it, which silently retypes
string data that happens to look like a Python literal — ``"211"`` becomes the
integer ``211``, ``"3"`` becomes ``3``, while ``"007"`` and ``"F"`` survive
because they fail to parse.  Every caller here renders such values into equality
filters and record fields, where a retyped value does not raise: it simply stops
matching a string column, and a workflow scoped by it reports an empty result as
though there were nothing to find.

``StrictUndefined`` is deliberate: a template naming something the context does
not have raises rather than rendering an empty string.  Both callers use these
templates for identifiers and provenance that are matched by exact equality
downstream, where a silent empty value is worse than an error.

Lives here rather than under ``workflows/`` because ``pipeline/`` renders the
same templates for the ``extract-structured`` ``fields:`` overlay, and a pipeline
that imports from ``workflows`` to do it would have the layering backwards.
``cogbase.workflows.context`` re-exports it, so existing imports keep working.

jinja2 arrives with the ``[api]`` extra rather than the base install, so callers
in the core package must reach ``render_value`` only on a path the user opted
into by writing a template — see ``jinja_available``.
"""

from __future__ import annotations

from typing import Any

try:
    from jinja2 import StrictUndefined, is_undefined
    from jinja2.nativetypes import NativeEnvironment, NativeTemplate, native_concat

    def _concat(values: Any) -> Any:
        """``native_concat``, minus the ``literal_eval`` on a single node.

        A template of exactly one ``{{ expr }}`` yields one node, and that node is
        already the object the context held — there is nothing to parse, so it is
        returned untouched.  Multi-node templates keep Jinja's behaviour: they are
        genuinely text being assembled, and anything that survives ``literal_eval``
        there was written as a literal by the config author.
        """
        nodes = list(values)
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]
        return native_concat(nodes)

    class _Environment(NativeEnvironment):
        concat = staticmethod(_concat)  # type: ignore[assignment]

    class _Template(NativeTemplate):
        # NativeTemplate.render reaches for ``environment_class.concat`` rather
        # than the instance's environment, so overriding ``concat`` alone is not
        # enough — the template class has to point back at the subclass.
        environment_class = _Environment

    _Environment.template_class = _Template
    _env = _Environment(undefined=StrictUndefined)
except ImportError:  # pragma: no cover
    _env = None  # type: ignore[assignment]

    def is_undefined(obj: Any) -> bool:
        return False

_MISSING = (
    "jinja2 is required for template rendering: pip install 'cogbase[api]'"
)


def jinja_available() -> bool:
    """Whether templates can be rendered in this install.

    Lets a caller fail at construction time, with a config-shaped message, rather
    than partway through the work that first needs a template.
    """
    return _env is not None


def render_value(value: Any, ctx: dict) -> Any:
    """Render *value* recursively against *ctx*.

    - ``str`` values are treated as Jinja2 templates.
    - ``list`` / ``dict`` values are recursed into element-by-element.
    - All other values are returned unchanged.
    """
    if _env is None:
        raise RuntimeError(_MISSING)
    if isinstance(value, str):
        return _env.from_string(value).render(**ctx)
    if isinstance(value, list):
        return [render_value(item, ctx) for item in value]
    if isinstance(value, dict):
        return {k: render_value(v, ctx) for k, v in value.items()}
    return value


def render_defined(template: str, ctx: dict, *, what: str) -> Any:
    """``render_value`` for a single template, refusing to return an undefined.

    ``StrictUndefined`` raises when an undefined is *used* — stringified, compared,
    iterated. A ``NativeEnvironment`` template that is one bare ``{{ expr }}``
    returns the node raw and never uses it, so a name the context lacks comes back
    as an ``Undefined`` object rather than raising. Left alone that object is
    written into a record, which is the silent-wrong-value failure these templates
    exist to prevent.

    *what* names the caller's field in the error, since the template alone rarely
    says which column it was filling.

    Note this is deliberately not the behaviour of ``render_value`` itself: the
    workflow ``structured-save`` overlay has rendered leniently since it shipped,
    and tightening it belongs in its own change rather than riding along with a
    new feature.
    """
    rendered = render_value(template, ctx)
    if is_undefined(rendered):
        raise ValueError(
            f"{what}: template {template!r} resolved to undefined. Add a default "
            "(e.g. `| default(None, true)`) if the value is genuinely optional."
        )
    return rendered
