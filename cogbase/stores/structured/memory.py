"""In-memory implementation of StructuredStoreBase."""

from pydantic import BaseModel

from cogbase.stores.base import StructuredStoreBase
from cogbase.stores.filters import Filter, matches
from cogbase.stores.schema import CollectionSchema, FieldType


class InMemoryStructuredStore(StructuredStoreBase):
    """Thread-unsafe in-memory store backed by plain dicts.

    All filtering is done in Python via the filter DSL regardless of field type.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, CollectionSchema] = {}
        # collection → { id_value → record_dict }
        self._records: dict[str, dict[str, dict]] = {}

    async def create_collection(self, schema: CollectionSchema) -> None:
        if schema.name in self._schemas:
            return  # idempotent
        self._schemas[schema.name] = schema
        self._records[schema.name] = {}

    async def save(self, collection: str, records: list[BaseModel]) -> None:
        schema = self._get_schema(collection)
        store = self._records[collection]
        for record in records:
            row = _serialize(record, schema)
            store[row[schema.id_field]] = row

    async def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]:
        self._get_schema(collection)
        fs = filters or []
        return [r for r in self._records[collection].values() if matches(r, fs)]

    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None:
        schema = self._get_schema(collection)
        store = self._records[collection]
        fs = filters or []
        if not fs:
            store.clear()
            return
        to_delete = [r[schema.id_field] for r in store.values() if matches(r, fs)]
        for key in to_delete:
            del store[key]

    def _get_schema(self, collection: str) -> CollectionSchema:
        if collection not in self._schemas:
            raise KeyError(f"Collection '{collection}' not found. Call create_collection first.")
        return self._schemas[collection]


def _serialize(record: BaseModel, schema: CollectionSchema) -> dict:
    raw = record.model_dump(mode="python")
    row: dict = {}
    for name, field in schema.fields.items():
        val = raw.get(name)
        if val is None:
            row[name] = None
            continue
        if field.type == FieldType.BOOLEAN:
            row[name] = bool(val)
        elif field.type == FieldType.INTEGER:
            row[name] = int(val)
        elif field.type == FieldType.FLOAT:
            row[name] = float(val)
        else:
            row[name] = val  # STRING and JSON stored as-is
    return row
