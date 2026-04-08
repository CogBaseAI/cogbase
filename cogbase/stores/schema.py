"""Schema types for defining structured store collections."""

from enum import Enum

from pydantic import BaseModel, field_validator, model_validator


class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"  # serialised as a JSON blob; filtered in Python, not SQL


class FieldSchema(BaseModel):
    type: FieldType
    nullable: bool = True
    index: bool = False  # create a DB index on this column (ignored by in-memory store)


class CollectionSchema(BaseModel):
    """Schema for a structured store collection (table).

    Args:
        name:     Collection name — must be a valid identifier (``[a-zA-Z_][a-zA-Z0-9_]*``).
        id_field: Name of the primary key field; must be present in ``fields``.
        fields:   Ordered mapping of field name → field schema.
                  Fields not listed here are silently dropped on save.
    """

    name: str
    id_field: str
    fields: dict[str, FieldSchema]

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", v):
            raise ValueError(
                f"Collection name '{v}' is invalid — use letters, digits, and underscores only"
            )
        return v

    @model_validator(mode="after")
    def _id_field_in_fields(self) -> "CollectionSchema":
        if self.id_field not in self.fields:
            raise ValueError(
                f"id_field '{self.id_field}' must be present in fields"
            )
        return self
