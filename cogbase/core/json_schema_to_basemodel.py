"""Build a Pydantic BaseModel class from a standard JSON Schema dict.

Supports JSON Schema draft-07 as produced by ``BaseModel.model_json_schema()``.

Handled constructs
------------------
- Scalar types: ``string``, ``integer``, ``number``, ``boolean``
- Optional scalars: ``anyOf: [{type: X}, {type: null}]``
- Nested objects via ``$ref`` or inline ``{type: object, properties: {...}}``
- Arrays with ``$ref`` items or primitive items
- ``$defs`` / ``definitions`` blocks for nested-model resolution
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, create_model


# Sentinel for "no default provided" — distinguishes from an explicit ``None`` default.
_MISSING: object = object()

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_model_from_json_schema(
    schema: dict | str,
    model_name: str = "DynamicModel",
) -> type[BaseModel]:
    """Build a Pydantic BaseModel class from a JSON Schema dict or JSON string.

    Args:
        schema:     JSON Schema dict **or** a JSON-encoded string (e.g. the
                    output of ``SomeModel.model_json_schema()``).
        model_name: Class name used for the root model when ``title`` is absent.

    Returns:
        A new ``BaseModel`` subclass whose fields mirror the schema.

    Example::

        from example.contract_analyst_demo.schema import ContractExtraction
        schema = ContractExtraction.model_json_schema()
        DynModel = build_model_from_json_schema(schema, model_name="ContractExtraction")
        record = DynModel.model_validate_json(llm_output)
    """
    if isinstance(schema, str):
        schema = json.loads(schema)

    defs: dict[str, dict] = schema.get("$defs", schema.get("definitions", {}))
    name: str = schema.get("title", model_name)
    return _build_object_model(schema, name, defs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_ref(ref: str, defs: dict[str, dict]) -> dict:
    """Resolve a ``$ref`` string (e.g. ``"#/$defs/Party"``) against *defs*."""
    key = ref.split("/")[-1]
    if key not in defs:
        raise ValueError(f"$ref '{ref}' not found in schema $defs")
    return defs[key]


def _unwrap_nullable(
    field_schema: dict, defs: dict[str, dict]
) -> tuple[dict, bool]:
    """Normalise nullable field schemas to ``(<inner schema>, is_nullable)``.

    Recognises both nullable forms allowed by JSON Schema:

    * **Type array** — ``"type": ["string", "null"]``
      (draft-04+, simpler hand-authored schemas)
    * **anyOf union** — ``"anyOf": [{"type": "string"}, {"type": "null"}]``
      (Pydantic v2 ``model_json_schema()`` output)

    Returns *(field_schema, False)* unchanged for non-nullable fields.
    """
    # --- type-array form: "type": ["string", "null"] ----------------------
    raw_type = field_schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t not in ("null", None)]
        has_null = len(non_null) < len(raw_type)
        if len(non_null) == 1:
            inner = {**field_schema, "type": non_null[0]}
            return inner, has_null
        # Multiple non-null types — fall back to str
        return {"type": "string"}, False

    # --- anyOf form: "anyOf": [{"type": "string"}, {"type": "null"}] ------
    any_of = field_schema.get("anyOf")
    if not any_of:
        return field_schema, False

    non_null = [s for s in any_of if s.get("type") not in ("null", None)]
    has_null = len(non_null) < len(any_of)

    if len(non_null) == 1:
        inner = non_null[0]
        if "$ref" in inner:
            inner = _resolve_ref(inner["$ref"], defs)
        return inner, has_null

    # Multiple non-null alternatives — fall back to str
    return {"type": "string"}, False


def _is_list_type(t: Any) -> bool:
    """Return ``True`` when *t* is a ``list[…]`` generic alias."""
    return getattr(t, "__origin__", None) is list


def _field_python_type(
    field_schema: dict,
    field_name: str,
    defs: dict[str, dict],
    is_nullable: bool,
) -> Any:
    """Map a resolved (non-anyOf) JSON Schema fragment to a Python type."""
    if "$ref" in field_schema:
        field_schema = _resolve_ref(field_schema["$ref"], defs)

    json_type = field_schema.get("type")

    if json_type in _JSON_TYPE_MAP:
        t: Any = _JSON_TYPE_MAP[json_type]
        return t | None if is_nullable else t  # type: ignore[return-value]

    if json_type == "array":
        items_schema = field_schema.get("items", {})
        if "$ref" in items_schema:
            items_schema = _resolve_ref(items_schema["$ref"], defs)

        if items_schema.get("type") == "object" or "properties" in items_schema:
            item_cls = _build_object_model(
                items_schema,
                items_schema.get("title", f"{field_name.capitalize()}Item"),
                defs,
            )
            return list[item_cls]  # type: ignore[valid-type]

        inner_type: type = _JSON_TYPE_MAP.get(items_schema.get("type", "string"), str)
        return list[inner_type]  # type: ignore[valid-type]

    if json_type == "object" or "properties" in field_schema:
        nested = _build_object_model(
            field_schema,
            field_schema.get("title", field_name.capitalize()),
            defs,
        )
        return nested | None if is_nullable else nested  # type: ignore[return-value]

    # Unknown / unsupported — fall back to Optional[str]
    return str | None if is_nullable else str  # type: ignore[return-value]


def _build_object_model(
    schema: dict,
    model_name: str,
    defs: dict[str, dict],
) -> type[BaseModel]:
    """Recursively build a ``BaseModel`` subclass for a JSON Schema object."""
    properties: dict[str, dict] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    field_defs: dict[str, Any] = {}

    for field_name, raw_schema in properties.items():
        description: str | None = raw_schema.get("description")
        default_val: Any = raw_schema.get("default", _MISSING)

        resolved_schema, is_nullable = _unwrap_nullable(raw_schema, defs)

        # Arrays are never nullable in the default CogBase schema pattern
        if resolved_schema.get("type") == "array":
            is_nullable = False

        py_type = _field_python_type(resolved_schema, field_name, defs, is_nullable)

        # Build Field kwargs
        if default_val is not _MISSING:
            if isinstance(default_val, list) and len(default_val) == 0:
                field_kwargs: dict[str, Any] = {"default_factory": list}
            elif isinstance(default_val, dict) and len(default_val) == 0:
                field_kwargs = {"default_factory": dict}
            else:
                field_kwargs = {"default": default_val}
        elif field_name in required:
            field_kwargs = {"default": ...}
        elif is_nullable:
            field_kwargs = {"default": None}
        elif _is_list_type(py_type):
            field_kwargs = {"default_factory": list}
        else:
            field_kwargs = {"default": None}

        if description:
            field_kwargs["description"] = description

        field_defs[field_name] = (py_type, Field(**field_kwargs))

    return create_model(model_name, **field_defs)
