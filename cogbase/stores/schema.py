"""Schema types for defining structured store collections."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"  # JSONB in Postgres (sub-key filters via dot notation); TEXT blob in SQLite (Python post-filter)


class FieldSchema(BaseModel):
    type: FieldType
    nullable: bool = True
    index: bool = False  # create a DB index on this column (ignored by in-memory store)
    unique: bool = False  # enforce a unique constraint on this column across all backends
    json_schema: str | None = None

    @model_validator(mode="after")
    def _json_schema_only_on_json_fields(self) -> "FieldSchema":
        if self.json_schema is not None and self.type != FieldType.JSON:
            raise ValueError("json_schema is only valid for FieldType.JSON fields")
        return self


class CollectionSchema(BaseModel):
    """Schema for a structured store collection (table).

    Args:
        name:           Collection name — must be a valid identifier
                        (``[a-zA-Z_][a-zA-Z0-9_]*``).
        primary_fields: Ordered list of primary-key field names; each must be
                        present in ``fields``.
        fields:         Ordered mapping of field name → field schema.
                        Fields not listed here are silently dropped on save.
        description:    Short description shown to the LLM in the retrieval system
                        prompt so it understands what this collection holds and when
                        to query it (e.g. "Extracted contract metadata: parties, dates,
                        governing law, termination clauses").
    """

    name: str
    primary_fields: list[str] = Field(min_length=1)
    fields: dict[str, FieldSchema]
    description: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", v):
            raise ValueError(
                f"Collection name '{v}' is invalid — use letters, digits, and underscores only"
            )
        return v

    @field_validator("primary_fields")
    @classmethod
    def _valid_primary_fields(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("primary_fields must not contain duplicates")
        return values

    @model_validator(mode="after")
    def _primary_fields_in_fields(self) -> "CollectionSchema":
        missing = [field for field in self.primary_fields if field not in self.fields]
        if missing:
            raise ValueError(
                f"primary_fields {missing!r} must be present in fields"
            )
        return self
