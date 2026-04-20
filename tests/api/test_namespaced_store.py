"""Unit tests for api/namespaced_store.py — NamespacedStructuredStore."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured.memory import InMemoryStructuredStore
from api.namespaced_store import NamespacedStructuredStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = CollectionSchema(
    name="items",
    primary_fields=["item_id"],
    fields={
        "item_id": FieldSchema(type=FieldType.STRING, nullable=False),
        "value":   FieldSchema(type=FieldType.STRING, nullable=True),
    },
)


class Item(BaseModel):
    item_id: str
    value: str | None = None


def _make_ns(namespace: str) -> tuple[InMemoryStructuredStore, NamespacedStructuredStore]:
    backend = InMemoryStructuredStore()
    ns = NamespacedStructuredStore(backend, namespace)
    return backend, ns


# ---------------------------------------------------------------------------
# Prefix sanitization
# ---------------------------------------------------------------------------

class TestPrefixSanitization:
    def test_alphanumeric_preserved(self):
        _, ns = _make_ns("myApp123")
        assert ns._prefix == "myApp123"

    def test_hyphens_replaced_with_underscore(self):
        _, ns = _make_ns("my-app")
        assert ns._prefix == "my_app"

    def test_spaces_replaced_with_underscore(self):
        _, ns = _make_ns("my app name")
        assert ns._prefix == "my_app_name"

    def test_mixed_special_chars(self):
        _, ns = _make_ns("app.v2!#")
        assert ns._prefix == "app_v2__"

    def test_ns_format(self):
        _, ns = _make_ns("myapp")
        assert ns._ns("contracts") == "myapp__contracts"


# ---------------------------------------------------------------------------
# create_collection
# ---------------------------------------------------------------------------

class TestCreateCollection:
    @pytest.mark.asyncio
    async def test_creates_namespaced_collection_in_backend(self):
        backend, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        # Backend should know "app1__items", not "items"
        await backend.save("app1__items", [Item(item_id="x")])
        rows = await backend.query("app1__items")
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_original_schema_name_unchanged(self):
        backend, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        # Original schema object must not be mutated
        assert SIMPLE_SCHEMA.name == "items"

    @pytest.mark.asyncio
    async def test_idempotent(self):
        _, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        await ns.create_collection(SIMPLE_SCHEMA)  # must not raise


# ---------------------------------------------------------------------------
# save and query
# ---------------------------------------------------------------------------

class TestSaveAndQuery:
    @pytest.mark.asyncio
    async def test_save_and_query_roundtrip(self):
        _, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        await ns.save("items", [Item(item_id="i1", value="hello")])
        rows = await ns.query("items")
        assert len(rows) == 1
        assert rows[0]["item_id"] == "i1"
        assert rows[0]["value"] == "hello"

    @pytest.mark.asyncio
    async def test_query_with_filters(self):
        from cogbase.stores.filters import Col
        _, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        await ns.save("items", [
            Item(item_id="i1", value="alpha"),
            Item(item_id="i2", value="beta"),
        ])
        rows = await ns.query("items", filters=[Col("item_id") == "i2"])
        assert len(rows) == 1
        assert rows[0]["value"] == "beta"

    @pytest.mark.asyncio
    async def test_query_with_field_projection(self):
        _, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        await ns.save("items", [Item(item_id="i1", value="hello")])
        rows = await ns.query("items", fields=["item_id"])
        assert "item_id" in rows[0]
        assert "value" not in rows[0]


# ---------------------------------------------------------------------------
# delete_records
# ---------------------------------------------------------------------------

class TestDeleteRecords:
    @pytest.mark.asyncio
    async def test_delete_all(self):
        _, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        await ns.save("items", [Item(item_id="i1"), Item(item_id="i2")])
        await ns.delete_records("items")
        assert await ns.query("items") == []

    @pytest.mark.asyncio
    async def test_delete_with_filter(self):
        from cogbase.stores.filters import Col
        _, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        await ns.save("items", [Item(item_id="i1"), Item(item_id="i2")])
        await ns.delete_records("items", filters=[Col("item_id") == "i1"])
        rows = await ns.query("items")
        assert len(rows) == 1
        assert rows[0]["item_id"] == "i2"


# ---------------------------------------------------------------------------
# update_collection
# ---------------------------------------------------------------------------

class TestUpdateCollection:
    @pytest.mark.asyncio
    async def test_update_collection_adds_field(self):
        backend, ns = _make_ns("app1")
        await ns.create_collection(SIMPLE_SCHEMA)
        await ns.save("items", [Item(item_id="i1", value="v")])

        extended_schema = CollectionSchema(
            name="items",
            primary_fields=["item_id"],
            fields={
                "item_id": FieldSchema(type=FieldType.STRING, nullable=False),
                "value":   FieldSchema(type=FieldType.STRING, nullable=True),
                "extra":   FieldSchema(type=FieldType.STRING, nullable=True),
            },
        )
        await ns.update_collection(extended_schema)
        rows = await ns.query("items")
        assert "extra" in rows[0]


# ---------------------------------------------------------------------------
# Isolation between two namespaces sharing one backend
# ---------------------------------------------------------------------------

class TestIsolation:
    @pytest.mark.asyncio
    async def test_two_apps_do_not_see_each_others_data(self):
        backend = InMemoryStructuredStore()
        ns1 = NamespacedStructuredStore(backend, "app-one")
        ns2 = NamespacedStructuredStore(backend, "app-two")

        await ns1.create_collection(SIMPLE_SCHEMA)
        await ns2.create_collection(SIMPLE_SCHEMA)

        await ns1.save("items", [Item(item_id="shared-id", value="from-app1")])
        await ns2.save("items", [Item(item_id="shared-id", value="from-app2")])

        r1 = await ns1.query("items")
        r2 = await ns2.query("items")

        assert r1[0]["value"] == "from-app1"
        assert r2[0]["value"] == "from-app2"

    @pytest.mark.asyncio
    async def test_delete_in_one_namespace_does_not_affect_other(self):
        backend = InMemoryStructuredStore()
        ns1 = NamespacedStructuredStore(backend, "app-one")
        ns2 = NamespacedStructuredStore(backend, "app-two")

        await ns1.create_collection(SIMPLE_SCHEMA)
        await ns2.create_collection(SIMPLE_SCHEMA)

        await ns1.save("items", [Item(item_id="i1", value="v1")])
        await ns2.save("items", [Item(item_id="i1", value="v2")])

        await ns1.delete_records("items")

        assert await ns1.query("items") == []
        assert len(await ns2.query("items")) == 1
