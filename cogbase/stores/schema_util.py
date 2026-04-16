import types as _types
from pydantic import BaseModel
from typing import List, Union, get_origin, get_args, Type, Dict

from cogbase.stores.schema import FieldSchema, FieldType


def _unwrap_optional(t):
    """Unwrap ``Optional[X]`` / ``X | None`` to ``X``; return *t* unchanged otherwise."""
    origin = get_origin(t)
    # typing.Optional / typing.Union
    if origin is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    # Python 3.10+ ``X | Y`` syntax (types.UnionType)
    if hasattr(_types, "UnionType") and isinstance(t, _types.UnionType):
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


def _json_schema_for_type(t) -> str | None:
    origin = get_origin(t)

    if origin in (list, List):
        inner = _unwrap_optional(get_args(t)[0])
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return f"[{cls_json_schema_for_llm(inner)}]"
        return None

    if isinstance(t, type) and issubclass(t, BaseModel):
        return cls_json_schema_for_llm(t)

    return None


def cls_generate_schema(cls: Type[BaseModel]) -> Dict[str, FieldSchema]:
    """
    Automatically generate a schema for a Pydantic model.
    str -> FieldType.STRING, list or nested BaseModel -> FieldType.JSON.
    Optional[X] / X | None is unwrapped to X before type matching.
    Nested BaseModel JSON fields carry LLM-facing ``json_schema`` metadata.
    """
    schema = {}
    for field_name, field_info in cls.model_fields.items():
        field_type = _unwrap_optional(field_info.annotation)
        origin = get_origin(field_type)

        # Primitive types
        if field_type is int:
            schema[field_name] = FieldSchema(type=FieldType.INTEGER)
        elif field_type is float:
            schema[field_name] = FieldSchema(type=FieldType.FLOAT)
        elif field_type is bool:
            schema[field_name] = FieldSchema(type=FieldType.BOOLEAN)
        elif field_type is str:
            schema[field_name] = FieldSchema(type=FieldType.STRING)
        elif origin in (list, List) or (isinstance(field_type, type) and issubclass(field_type, BaseModel)):
            schema[field_name] = FieldSchema(
                type=FieldType.JSON,
                json_schema=_json_schema_for_type(field_type),
            )
        else:
            schema[field_name] = FieldSchema(type=FieldType.STRING)
    return schema


def type_to_str(t):
    origin = get_origin(t)

    if origin in (list, List):
        inner = get_args(t)[0]
        return f"List[{type_to_str(inner)}]"
    elif isinstance(t, type):
        if t is str:
            return "string"
        return t.__name__
    return str(t)


def cls_json_schema_for_llm(cls: Type[BaseModel], indent: int = 2) -> str:
    prefix = " " * indent
    lines = ["{"]

    for i, (field_name, field_info) in enumerate(cls.model_fields.items()):
        field_type = field_info.annotation
        origin = get_origin(field_type)
        desc = field_info.description or ""
        type_str = type_to_str(field_type)

        # List of BaseModel
        if origin in (list, List):
            inner_type = get_args(field_type)[0]
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                nested = cls_json_schema_for_llm(inner_type, indent + 2)
                lines.append(f'{prefix}"{field_name}": [{nested}],')
            else:
                lines.append(f'{prefix}"{field_name}": ["{desc}"],')
        # Nested BaseModel
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested = cls_json_schema_for_llm(field_type, indent + 2)
            lines.append(f'{prefix}"{field_name}": {nested},')
        # Primitive
        else:
            if desc != "":
                lines.append(f'{prefix}"{field_name}": "{type_str}, {desc}",')
            else:
                lines.append(f'{prefix}"{field_name}": "{type_str}",')

    lines[-1] = lines[-1].rstrip(",")
    lines.append(" " * (indent - 2) + "}")
    return "\n".join(lines)
