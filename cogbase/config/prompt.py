"""Shared helpers for rendering config models into YAML prompt templates."""

from __future__ import annotations

import re
import sys
from typing import Any, ForwardRef, Literal, get_args, get_origin, get_type_hints

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


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin is None:
        return annotation, False
    args = get_args(annotation)
    if type(None) in args and len(args) == 2:
        inner = next(arg for arg in args if arg is not type(None))
        return inner, True
    return annotation, False


def _unwrap_list(annotation: Any) -> Any | None:
    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        if args:
            return args[0]
    return None


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


def _should_skip_field(field: Any) -> bool:
    extra = getattr(field, "json_schema_extra", None) or {}
    return bool(extra.get("prompt_skip") or extra.get("prompt_exclude"))


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
    for field_name, field in model_cls.model_fields.items():
        if _should_skip_field(field):
            continue
        alias = field.alias or field_name
        annotation, optional = _unwrap_optional(type_hints.get(field_name, field.annotation))
        description = field.description
        default = _resolved_default(field)

        list_item = _unwrap_list(annotation)
        if list_item is not None:
            list_item = _resolve_model_type(list_item, model_cls)
            if isinstance(list_item, type) and issubclass(list_item, BaseModel):
                lines.append(f"{pad}{alias}:{_field_comment(description, default, optional)}")
                if list_item in _stack:
                    lines.append(f"{' ' * (indent + 2)}- ...")
                else:
                    item_lines = _render_model_template(list_item, indent + 2, _stack + (model_cls,))
                    if item_lines:
                        item_lines[0] = f"{' ' * indent}- {item_lines[0].lstrip()}"
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
            lines.append(f"{pad}{alias}: {value}{_field_comment(description, default, optional)}")
            continue

        if default is not PydanticUndefined:
            value = _yaml_scalar(default)
        elif optional:
            value = "null"
        else:
            value = f"<{field_name}>"
        lines.append(f"{pad}{alias}: {value}{_field_comment(description, default, optional)}")

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
