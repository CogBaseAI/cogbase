"""Shared helpers for rendering config models into YAML prompt templates."""

from __future__ import annotations

import re
import sys
import types
from typing import Annotated, Any, ForwardRef, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if re.fullmatch(r"[A-Za-z0-9_./:-]+", value):
            return value
        return f'"{value}"'
    if isinstance(value, list):
        return "[]"
    if isinstance(value, dict):
        return "{}"
    return str(value)


def _literal_values(annotation: Any) -> list[Any]:
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    return []


def _single_literal_value(annotation: Any) -> Any | None:
    values = _literal_values(annotation)
    if len(values) == 1:
        return values[0]
    return None


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    annotation = _strip_annotated(annotation)
    origin = get_origin(annotation)
    if origin is None:
        return annotation, False
    args = get_args(annotation)
    if type(None) in args and len(args) == 2:
        inner = next(arg for arg in args if arg is not type(None))
        return inner, True
    return annotation, False


def _unwrap_list(annotation: Any) -> Any | None:
    annotation = _strip_annotated(annotation)
    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        if args:
            return args[0]
    return None


def _strip_annotated(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        return get_args(annotation)[0]
    return annotation


def _union_variants(annotation: Any) -> list[Any]:
    annotation = _strip_annotated(annotation)
    origin = get_origin(annotation)
    if origin not in (types.UnionType, Union):
        return []
    return list(get_args(annotation))


def _resolve_base_model_variants(annotation: Any, model_cls: type[BaseModel]) -> list[type[BaseModel]]:
    variants: list[type[BaseModel]] = []
    for variant in _union_variants(annotation):
        if variant is type(None):
            continue
        variant = _resolve_model_type(variant, model_cls)
        if isinstance(variant, type) and issubclass(variant, BaseModel):
            variants.append(variant)
    return variants


def _resolve_model_type(annotation: Any, model_cls: type[BaseModel]) -> Any:
    if isinstance(annotation, type):
        return annotation
    if isinstance(annotation, ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        module = sys.modules.get(model_cls.__module__)
        if module is not None:
            return getattr(module, annotation, annotation)
    return annotation


def _field_comment(description: str | None, default: Any = PydanticUndefined, optional: bool = False) -> str:
    parts: list[str] = []
    if description:
        parts.append(description)
    if optional:
        parts.append("optional")
    if default is not PydanticUndefined and default is not None and not isinstance(default, BaseModel):
        parts.append(f"default: {_yaml_scalar(default)}")
    if not parts:
        return ""
    return "  # " + "; ".join(parts)


def _resolved_default(field: Any) -> Any:
    default = field.default
    if default is not PydanticUndefined:
        return default
    default_factory = getattr(field, "default_factory", None)
    if default_factory is None:
        return PydanticUndefined
    try:
        value = field.get_default(call_default_factory=True)
    except TypeError:
        return PydanticUndefined
    if isinstance(value, BaseModel):
        return PydanticUndefined
    return value


def _should_skip_field(field: Any, annotation: Any) -> bool:
    extra = getattr(field, "json_schema_extra", None) or {}
    return bool(extra.get("prompt_skip") or extra.get("prompt_exclude"))


def _field_comment_for_annotation(
    description: str | None,
    annotation: Any,
    default: Any = PydanticUndefined,
    optional: bool = False,
) -> str:
    if _single_literal_value(annotation) is not None:
        default = PydanticUndefined
    return _field_comment(description, default, optional)


def _discriminator_label(model_cls: type[BaseModel], field_name: str = "tool") -> str | None:
    field = model_cls.model_fields.get(field_name)
    if field is None:
        return None
    type_hints = get_type_hints(model_cls, include_extras=True)
    annotation, _ = _unwrap_optional(type_hints.get(field_name, field.annotation))
    value = _single_literal_value(annotation)
    if value is None:
        return None
    return str(value)


def _render_model_template(
    model_cls: type[BaseModel],
    indent: int = 0,
    _stack: tuple[type[BaseModel], ...] = (),
    *,
    first_line_prefix: str = "",
) -> list[str]:
    """Render a BaseModel class as YAML-shaped lines with inline descriptions."""

    pad = " " * indent
    lines: list[str] = []
    type_hints = get_type_hints(model_cls, include_extras=True)
    field_items = list(model_cls.model_fields.items())
    if "tool" in model_cls.model_fields:
        field_items = [("tool", model_cls.model_fields["tool"])] + [
            (name, field) for name, field in field_items if name != "tool"
        ]
    for field_name, field in field_items:
        alias = field.alias or field_name
        annotation, optional = _unwrap_optional(type_hints.get(field_name, field.annotation))
        if _should_skip_field(field, annotation):
            continue
        description = field.description
        default = _resolved_default(field)

        list_item = _unwrap_list(annotation)
        if list_item is not None:
            model_variants = _resolve_base_model_variants(list_item, model_cls)
            if model_variants:
                lines.append(f"{pad}{alias}:{_field_comment(description, default, optional)}")
                for variant in model_variants:
                    if variant in _stack:
                        lines.append(f"{' ' * (indent + 2)}- ...")
                        continue
                    item_lines = _render_model_template(variant, indent + 4, _stack + (model_cls,))
                    if item_lines:
                        item_lines[0] = f"{' ' * (indent + 2)}- {item_lines[0].lstrip()}"
                    lines.extend(item_lines)
            elif isinstance(list_item, type) and issubclass(list_item, BaseModel):
                lines.append(f"{pad}{alias}:{_field_comment(description, default, optional)}")
                if list_item in _stack:
                    lines.append(f"{' ' * (indent + 2)}- ...")
                else:
                    item_lines = _render_model_template(list_item, indent + 4, _stack + (model_cls,))
                    if item_lines:
                        item_lines[0] = f"{' ' * (indent + 2)}- {item_lines[0].lstrip()}"
                    lines.extend(item_lines)
            else:
                if default is PydanticUndefined or default is None or default == []:
                    item_value = "[<item>]"
                else:
                    item_value = _yaml_scalar(default)
                lines.append(f"{pad}{alias}: {item_value}{_field_comment(description, default, optional)}")
            continue

        annotation = _resolve_model_type(annotation, model_cls)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            lines.append(f"{pad}{alias}:{_field_comment(description, default, optional)}")
            if annotation in _stack:
                lines.append(f"{' ' * (indent + 2)}...")
            else:
                lines.extend(_render_model_template(annotation, indent + 2, _stack + (model_cls,)))
            continue

        literal_values = _literal_values(annotation)
        if literal_values:
            if default is not PydanticUndefined and default is not None:
                value = _yaml_scalar(default)
            else:
                value = " | ".join(_yaml_scalar(v) for v in literal_values)
            lines.append(f"{pad}{alias}: {value}{_field_comment_for_annotation(description, annotation, default, optional)}")
            continue

        if default is not PydanticUndefined:
            value = _yaml_scalar(default)
        elif optional:
            value = "null"
        else:
            value = f"<{field_name}>"
        lines.append(f"{pad}{alias}: {value}{_field_comment_for_annotation(description, annotation, default, optional)}")

    if lines and first_line_prefix:
        lines[0] = f"{pad}{first_line_prefix}{lines[0].lstrip()}"
    return lines


def render_config_template(model_cls: type[BaseModel], indent: int = 0, _stack: tuple[type[BaseModel], ...] = ()) -> str:
    return "\n".join(_render_model_template(model_cls, indent, _stack))


class ConfigPromptMixin:
    """Mixin that exposes a generated YAML template for a config model."""

    @classmethod
    def config_format_prompt(cls) -> str:
        return render_config_template(cls)
