"""Unit tests for api/factory.py — build_structured_store and build_app."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.config.config import AppConfig
from cogbase.config.stores import StructuredStoreConfig
from api.factory import build_app, _json_schema_to_collection_fields
from api.system_resources import SystemResources
from cogbase.stores import build_structured_store
from cogbase.stores.schema import FieldType
from cogbase.pipeline.ingestion_pipeline import VectorCollection, PipelineStep
from cogbase.stores.document.local_fs import LocalFSDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.structured.sqlite import SQLiteStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# build_structured_store
# ---------------------------------------------------------------------------

class TestBuildStructuredStore:
    def test_memory_type(self):
        cfg = StructuredStoreConfig(type="memory")
        store = build_structured_store(cfg)
        assert isinstance(store, InMemoryStructuredStore)

    def test_sqlite_type(self, tmp_path):
        cfg = StructuredStoreConfig(type="sqlite", path=str(tmp_path / "test.db"))
        store = build_structured_store(cfg)
        assert isinstance(store, SQLiteStructuredStore)
        store.close()

    def test_sqlite_in_memory(self):
        cfg = StructuredStoreConfig(type="sqlite", path=":memory:")
        store = build_structured_store(cfg)
        assert isinstance(store, SQLiteStructuredStore)
        store.close()


# ---------------------------------------------------------------------------
# _json_schema_to_collection_fields
# ---------------------------------------------------------------------------

class TestJsonSchemaToCollectionFields:
    def test_empty_schema_returns_empty_dict(self):
        assert _json_schema_to_collection_fields({}) == {}

    def test_empty_properties_returns_empty_dict(self):
        assert _json_schema_to_collection_fields({"type": "object", "properties": {}}) == {}

    def test_string_field(self):
        result = _json_schema_to_collection_fields({"properties": {"name": {"type": "string"}}})
        assert result["name"].type == FieldType.STRING

    def test_integer_field(self):
        result = _json_schema_to_collection_fields({"properties": {"count": {"type": "integer"}}})
        assert result["count"].type == FieldType.INTEGER

    def test_number_field(self):
        result = _json_schema_to_collection_fields({"properties": {"price": {"type": "number"}}})
        assert result["price"].type == FieldType.FLOAT

    def test_boolean_field(self):
        result = _json_schema_to_collection_fields({"properties": {"active": {"type": "boolean"}}})
        assert result["active"].type == FieldType.BOOLEAN

    def test_object_field(self):
        result = _json_schema_to_collection_fields({"properties": {"meta": {"type": "object"}}})
        assert result["meta"].type == FieldType.JSON

    def test_array_field(self):
        result = _json_schema_to_collection_fields({"properties": {"tags": {"type": "array"}}})
        assert result["tags"].type == FieldType.JSON

    def test_unknown_type_falls_back_to_string(self):
        result = _json_schema_to_collection_fields({"properties": {"x": {"type": "null"}}})
        assert result["x"].type == FieldType.STRING

    def test_missing_type_falls_back_to_string(self):
        result = _json_schema_to_collection_fields({"properties": {"x": {}}})
        assert result["x"].type == FieldType.STRING

    def test_anyof_nullable_unwraps_to_non_null_type(self):
        schema = {"properties": {"score": {"anyOf": [{"type": "number"}, {"type": "null"}]}}}
        result = _json_schema_to_collection_fields(schema)
        assert result["score"].type == FieldType.FLOAT

    def test_anyof_nullable_null_first(self):
        schema = {"properties": {"flag": {"anyOf": [{"type": "null"}, {"type": "boolean"}]}}}
        result = _json_schema_to_collection_fields(schema)
        assert result["flag"].type == FieldType.BOOLEAN

    def test_anyof_all_null_falls_back_to_string(self):
        schema = {"properties": {"x": {"anyOf": [{"type": "null"}, {"type": "null"}]}}}
        result = _json_schema_to_collection_fields(schema)
        assert result["x"].type == FieldType.STRING

    def test_multiple_fields_mixed_types(self):
        schema = {
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"},
                "details": {"type": "object"},
                "tags": {"type": "array"},
            }
        }
        result = _json_schema_to_collection_fields(schema)
        assert result["title"].type == FieldType.STRING
        assert result["year"].type == FieldType.INTEGER
        assert result["price"].type == FieldType.FLOAT
        assert result["active"].type == FieldType.BOOLEAN
        assert result["details"].type == FieldType.JSON
        assert result["tags"].type == FieldType.JSON

    def test_field_order_preserved(self):
        schema = {"properties": {"z": {"type": "string"}, "a": {"type": "integer"}}}
        result = _json_schema_to_collection_fields(schema)
        assert list(result.keys()) == ["z", "a"]

    def test_anyof_ref_and_null_maps_to_json(self):
        # Regression: PaymentTerms | None → anyOf[$ref, null] was incorrectly
        # mapped to STRING because $ref has no "type" key and was filtered out.
        schema = {
            "properties": {
                "payment_terms": {
                    "anyOf": [
                        {"$ref": "#/$defs/PaymentTerms"},
                        {"type": "null"},
                    ]
                }
            }
        }
        result = _json_schema_to_collection_fields(schema)
        assert result["payment_terms"].type == FieldType.JSON

    def test_anyof_ref_without_null_maps_to_json(self):
        schema = {
            "properties": {
                "nested": {"anyOf": [{"$ref": "#/$defs/SomeModel"}]}
            }
        }
        result = _json_schema_to_collection_fields(schema)
        assert result["nested"].type == FieldType.JSON

    def test_bare_ref_field_maps_to_json(self):
        schema = {"properties": {"nested": {"$ref": "#/$defs/SomeModel"}}}
        result = _json_schema_to_collection_fields(schema)
        assert result["nested"].type == FieldType.JSON


# ---------------------------------------------------------------------------
# Regression: nested-object (anyOf $ref | null) round-trip via SQLite
# ---------------------------------------------------------------------------

class TestNestedObjectSqliteRoundTrip:
    """Verify that $ref-typed fields survive a full SQLite save/query cycle.

    Before the fix, _json_schema_to_collection_fields mapped PaymentTerms | None
    to FieldType.STRING, causing sqlite3.InterfaceError when a dict value arrived.
    """

    async def test_ref_nullable_field_saves_and_queries(self):
        from pydantic import BaseModel, Field, create_model
        from cogbase.stores.structured.sqlite import SQLiteStructuredStore
        from cogbase.stores.schema import CollectionSchema

        class PaymentTerms(BaseModel):
            schedule: str | None = None
            due_date: str | None = None

        class ContractExtraction(BaseModel):
            contract_type: str | None = None
            payment_terms: PaymentTerms | None = None

        ContractRecord = create_model(
            "ContractRecord", doc_id=(str, ...), __base__=ContractExtraction
        )

        json_schema = ContractRecord.model_json_schema()
        fields = _json_schema_to_collection_fields(json_schema)

        assert fields["payment_terms"].type == FieldType.JSON

        schema = CollectionSchema(
            name="contracts",
            description="test",
            primary_fields=["doc_id"],
            fields=fields,
        )

        store = SQLiteStructuredStore(":memory:")
        await store.create_collection(schema)

        record = ContractRecord(
            doc_id="doc-1",
            contract_type="SaaS",
            payment_terms=PaymentTerms(schedule="net-30", due_date="2025-01-01"),
        )
        await store.save("contracts", [record.model_dump()])

        rows = await store.query("contracts")
        assert len(rows) == 1
        assert rows[0]["doc_id"] == "doc-1"
        assert rows[0]["payment_terms"] == {"schedule": "net-30", "due_date": "2025-01-01"}


# ---------------------------------------------------------------------------
# build_app — structured store resolution
# ---------------------------------------------------------------------------

_EXTRACTION_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'
_RECORD_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"},"doc_id":{"type":"string"}}}'
_LIST_EXTRACTION_SCHEMA = _EXTRACTION_SCHEMA
_LIST_RECORD_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"},"doc_id":{"type":"string"},"clause_id":{"type":"string"}}}'

_EXTRACT_ONLY_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
structured_collections:
  - name: contract_extraction
    description: Extracted contract facts and entities for exact lookup.
    schema: '{_RECORD_SCHEMA}'
    primary_fields: [doc_id]
pipelines:
  - name: main
    routing_description: Contract documents to extract structured facts from.
    steps:
      - tool: extract-structured
        collection: contract_extraction
        extractor:
          type: llm
          extraction_schema: '{_EXTRACTION_SCHEMA}'
          prompt: Extract contract facts.
"""

_FULL_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
vector_collections:
  - name: document_chunks
    description: Full-text document chunks for detailed retrieval.
structured_collections:
  - name: contract_extraction
    description: Extracted contract facts and entities for exact lookup.
    schema: '{_RECORD_SCHEMA}'
    primary_fields: [doc_id]
pipelines:
  - name: main
    routing_description: Contract documents for chunked indexing and structured extraction.
    steps:
      - tool: chunk-embed-upsert
        collection: document_chunks
        chunker:
          type: fixed
          chunk_size: 512
          overlap: 64
      - tool: extract-structured
        collection: contract_extraction
        extractor:
          type: llm
          extraction_schema: '{_EXTRACTION_SCHEMA}'
          prompt: Extract contract facts.
"""


def _mock_llm():
    from cogbase.llms.base import LLMBase
    llm = MagicMock(spec=LLMBase)
    llm.complete = AsyncMock(return_value="")
    async def _empty_stream(*a, **kw):
        return
        yield  # make it an async generator
    llm.complete_stream = _empty_stream
    return llm


class TestBuildAppStructuredStoreResolution:
    @patch("api.factory._build_llm")
    async def test_raises_when_no_stores(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_EXTRACT_ONLY_CONFIG_YAML)
        with pytest.raises(ValueError, match="structured store"):
            await build_app(cfg, app_status="initializing")

    @patch("api.factory._build_llm")
    async def test_uses_system_store_when_no_app_store(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_EXTRACT_ONLY_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )
        structured_store = next(iter(app._pipelines[0]._structured_by_name.values())).store
        assert structured_store is system_store

    @patch("api.factory._build_llm")
    async def test_app_config_store_overrides_system_store(self, mock_build_llm, tmp_path):
        mock_build_llm.return_value = _mock_llm()
        cfg_yaml = _EXTRACT_ONLY_CONFIG_YAML + (
            "structured_store:\n  type: sqlite\n  path: \":memory:\"\n"
        )
        cfg = AppConfig.from_yaml(cfg_yaml)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )
        structured_store = next(iter(app._pipelines[0]._structured_by_name.values())).store
        assert isinstance(structured_store, SQLiteStructuredStore)


class TestBuildAppCollectionCreation:
    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_creates_collections_for_new_app(self, mock_build_embedder, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        vector_store = MagicMock()
        vector_store.create_collection = AsyncMock()
        structured_store = MagicMock()
        structured_store.create_collection = AsyncMock()

        await build_app(
            cfg,
            system=SystemResources(structured_store=structured_store, vector_store=vector_store),
            app_status="initializing",
        )

        vector_store.create_collection.assert_awaited_once()
        structured_store.create_collection.assert_awaited_once()

    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_skips_collection_creation_for_active_restore(self, mock_build_embedder, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        vector_store = MagicMock()
        vector_store.create_collection = AsyncMock()
        structured_store = MagicMock()
        structured_store.create_collection = AsyncMock()

        await build_app(
            cfg,
            system=SystemResources(structured_store=structured_store, vector_store=vector_store),
            app_status="active",
        )

        vector_store.create_collection.assert_not_awaited()
        structured_store.create_collection.assert_not_awaited()

    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_register_schema_called_for_active_restore(self, mock_build_embedder, mock_build_llm):
        """Active restore must call register_schema so _schemas is populated without DDL."""
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        vector_store = MagicMock()
        vector_store.create_collection = AsyncMock()
        structured_store = MagicMock()
        structured_store.create_collection = AsyncMock()

        await build_app(
            cfg,
            system=SystemResources(structured_store=structured_store, vector_store=vector_store),
            app_status="active",
        )

        structured_store.register_schema.assert_called_once()
        called_schema = structured_store.register_schema.call_args[0][0]
        assert called_schema.name == "contract_extraction"

    @patch("api.factory._build_llm")
    async def test_active_restore_enables_query_on_real_store(self, mock_build_llm):
        """End-to-end: schemas registered by build_app (active) let the store serve queries."""
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType

        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_EXTRACT_ONLY_CONFIG_YAML)
        store = InMemoryStructuredStore()

        # First pass — initialising: creates the table.
        await build_app(cfg, system=SystemResources(structured_store=store), app_status="initializing")

        # Simulate restart: wipe only _schemas (the data frame persists in a real
        # backing store; InMemoryStructuredStore's frame is retained here to mimic
        # that a SQLite/Postgres table survives across process restarts).
        store._schemas.clear()

        # Second pass — active: must register schemas so query doesn't raise KeyError.
        await build_app(cfg, system=SystemResources(structured_store=store), app_status="active")

        # Query must succeed (empty result, no KeyError).
        results = await store.query("contract_extraction")
        assert results == []


class TestBuildAppVectorStoreResolution:
    @patch("api.factory._build_llm")
    async def test_no_vector_collection_when_no_chunk_step(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_EXTRACT_ONLY_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )
        assert app._pipelines[0]._vector_by_name == {}

    @patch("api.factory._build_llm")
    async def test_system_vector_store_used_when_chunk_step_present(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system=SystemResources(structured_store=system_store, vector_store=sys_vs),
                app_status="initializing",
            )

        assert app._pipelines[0]._vector_by_name

    @patch("api.factory._build_llm")
    async def test_vector_collection_name_matches_config(self, mock_build_llm):
        """The vector collection name comes from vector_collections config, not app name."""
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system=SystemResources(structured_store=system_store, vector_store=sys_vs),
                app_status="initializing",
            )

        assert "document_chunks" in app._pipelines[0]._vector_by_name


# ---------------------------------------------------------------------------
# build_app — document-embed-upsert step
# ---------------------------------------------------------------------------

_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'  # already defined above, reused

_SUMMARIZE_ONLY_CONFIG_YAML = """\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
vector_collections:
  - name: document_summary
    description: One summary vector per document for topic-level search.
pipelines:
  - name: main
    routing_description: Documents to index as per-document summaries.
    steps:
      - tool: document-embed-upsert
        collection: document_summary
        doc_prompt: "Summarize in one sentence."
"""

_THREE_STEP_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
vector_collections:
  - name: document_chunks
    description: Full-text document chunks for detailed retrieval.
  - name: document_summary
    description: One summary vector per document for topic-level search.
structured_collections:
  - name: contract_extraction
    description: Extracted contract facts and entities for exact lookup.
    schema: '{_RECORD_SCHEMA}'
    primary_fields: [doc_id]
pipelines:
  - name: main
    routing_description: Contract documents for full three-step ingestion.
    steps:
      - tool: chunk-embed-upsert
        collection: document_chunks
        chunker:
          type: fixed
          chunk_size: 512
          overlap: 64
      - tool: extract-structured
        collection: contract_extraction
        extractor:
          type: llm
          extraction_schema: '{_EXTRACTION_SCHEMA}'
          prompt: Extract contract facts.
      - tool: document-embed-upsert
        collection: document_summary
        doc_prompt: "Summarize in one sentence."
"""


class TestBuildAppDocumentCollection:
    @patch("api.factory._build_llm")
    async def test_document_collection_present_in_pipeline(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_SUMMARIZE_ONLY_CONFIG_YAML)
        sys_vs = FAISSVectorStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system=SystemResources(vector_store=sys_vs),
                app_status="initializing",
            )

        assert "document_summary" in app._pipelines[0]._vector_by_name

    @patch("api.factory._build_llm")
    async def test_document_step_prompt(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_SUMMARIZE_ONLY_CONFIG_YAML)
        sys_vs = FAISSVectorStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system=SystemResources(vector_store=sys_vs),
                app_status="initializing",
            )

        vc = app._pipelines[0]._vector_by_name["document_summary"]
        assert vc.name == "document_summary"
        step = next(s for s in app._pipelines[0]._steps if s.collection == "document_summary")
        assert step.doc_prompt == "Summarize in one sentence."

    @patch("api.factory._build_llm")
    async def test_all_vector_collections_share_vector_store(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_THREE_STEP_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system=SystemResources(structured_store=system_store, vector_store=sys_vs),
                app_status="initializing",
            )

        stores = {vc.store for vc in app._pipelines[0]._vector_by_name.values()}
        assert len(stores) == 1, "all vector collections should share the same store"

    @patch("api.factory._build_llm")
    async def test_three_step_pipeline_builds_all_collections(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_THREE_STEP_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system=SystemResources(structured_store=system_store, vector_store=sys_vs),
                app_status="initializing",
            )

        assert app._pipelines[0]._vector_by_name
        assert app._pipelines[0]._structured_by_name
        assert "document_summary" in app._pipelines[0]._vector_by_name

    @patch("api.factory._build_llm")
    async def test_vector_collection_names_includes_both(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_THREE_STEP_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system=SystemResources(structured_store=system_store, vector_store=sys_vs),
                app_status="initializing",
            )

        assert "document_chunks" in app._pipelines[0]._vector_by_name
        assert "document_summary" in app._pipelines[0]._vector_by_name

    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_summarize_step_without_vector_store_raises(self, mock_build_embedder, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        cfg = AppConfig.from_yaml(_SUMMARIZE_ONLY_CONFIG_YAML)
        # No vector store supplied
        with pytest.raises(ValueError, match="vector store"):
            await build_app(cfg, app_status="initializing")


# ---------------------------------------------------------------------------
# build_app — list extractor (extract_as_list / item_id_field)
# ---------------------------------------------------------------------------

_LIST_EXTRACTOR_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
structured_collections:
  - name: contract_clauses
    description: Extracted contract clauses with clause type and verbatim text.
    schema: '{_LIST_RECORD_SCHEMA}'
    primary_fields: [clause_id]
pipelines:
  - name: main
    routing_description: Contract documents to extract individual clauses from.
    steps:
      - tool: extract-structured
        collection: contract_clauses
        extractor:
          type: llm
          extraction_schema: '{_LIST_EXTRACTION_SCHEMA}'
          prompt: Extract contract clauses.
          record_mode: many
          response_field: clauses
          id_field: clause_id
          id_template: "{{doc_id}}__{{index:04d}}"
"""

_LIST_EXTRACTOR_WITH_PROMPT_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
structured_collections:
  - name: contract_clauses
    description: Extracted contract clauses with clause type and verbatim text.
    schema: '{_LIST_RECORD_SCHEMA}'
    primary_fields: [clause_id]
pipelines:
  - name: main
    routing_description: Contract documents to extract individual clauses with custom prompt.
    steps:
      - tool: extract-structured
        collection: contract_clauses
        extractor:
          type: llm
          extraction_schema: '{_LIST_EXTRACTION_SCHEMA}'
          record_mode: many
          response_field: clauses
          id_field: clause_id
          id_template: "{{doc_id}}__{{index:04d}}"
          prompt: "Extract all clauses.\\n\\n"
"""

_SINGLE_EXTRACTOR_WITH_PROMPT_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
structured_collections:
  - name: contract_metadata
    description: Extracted contract facts and entities for exact lookup.
    schema: '{_RECORD_SCHEMA}'
    primary_fields: [doc_id]
pipelines:
  - name: main
    routing_description: Contract documents to extract metadata fields from.
    steps:
      - tool: extract-structured
        collection: contract_metadata
        extractor:
          type: llm
          extraction_schema: '{_EXTRACTION_SCHEMA}'
          prompt: "Extract metadata.\\n\\n"
"""


class TestBuildAppListExtractor:
    def _get_extractor(self, app, collection: str):
        return next(s for s in app._pipelines[0]._steps if s.collection == collection).extractor

    @patch("api.factory._build_llm")
    async def test_list_extractor_record_mode_many(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_LIST_EXTRACTOR_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )

        extractor = self._get_extractor(app, "contract_clauses")
        assert extractor._record_mode == "many"

    @patch("api.factory._build_llm")
    async def test_list_extractor_custom_id_field_in_injected_fields(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_LIST_EXTRACTOR_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )

        extractor = self._get_extractor(app, "contract_clauses")
        assert "clause_id" in extractor._injected_fields

    @patch("api.factory._build_llm")
    async def test_list_extractor_custom_response_field(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_LIST_EXTRACTOR_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )

        extractor = self._get_extractor(app, "contract_clauses")
        assert extractor._response_field == "clauses"

    @patch("api.factory._build_llm")
    async def test_list_extractor_with_prompt_includes_response_field_instruction(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_LIST_EXTRACTOR_WITH_PROMPT_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )

        extractor = self._get_extractor(app, "contract_clauses")
        assert '"clauses"' in extractor._system_prompt
        assert "Extract all clauses." in extractor._system_prompt

    @patch("api.factory._build_llm")
    async def test_list_extractor_no_prompt_uses_default(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_LIST_EXTRACTOR_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )

        extractor = self._get_extractor(app, "contract_clauses")
        assert '"clauses"' in extractor._system_prompt

    @patch("api.factory._build_llm")
    async def test_single_extractor_with_prompt_does_not_include_list_instruction(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_SINGLE_EXTRACTOR_WITH_PROMPT_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )

        extractor = self._get_extractor(app, "contract_metadata")
        assert extractor._record_mode == "one"
        assert "Extract metadata." in extractor._system_prompt
        assert "whose value is an array" not in extractor._system_prompt

    @patch("api.factory._build_llm")
    async def test_single_extractor_has_only_doc_id_injected(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_EXTRACT_ONLY_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store),
            app_status="initializing",
        )

        extractor = self._get_extractor(app, "contract_extraction")
        assert extractor._record_mode == "one"
        assert list(extractor._injected_fields.keys()) == ["doc_id"]


class TestBuildAppDocumentStoreResolution:
    @patch("api.factory._build_llm")
    async def test_uses_system_document_store_when_no_app_document_store(self, mock_build_llm, tmp_path):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml("""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
""")
        sys_doc = LocalFSDocumentStore(tmp_path / "docs")
        app = await build_app(
            cfg,
            system=SystemResources(document_store=sys_doc),
            app_status="initializing",
        )

        assert app.document_store is sys_doc

    @patch("api.factory._build_llm")
    async def test_app_document_store_overrides_system_document_store(self, mock_build_llm, tmp_path):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
document_store:
  type: local
  path: {tmp_path / "app-docs"}
""")
        sys_doc = LocalFSDocumentStore(tmp_path / "system-docs")
        app = await build_app(
            cfg,
            system=SystemResources(document_store=sys_doc),
            app_status="initializing",
        )

        assert isinstance(app.document_store, LocalFSDocumentStore)
        assert app.document_store._root.name == "app-docs"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# build_app — routing match keys auto-injected into metadata_fields
# ---------------------------------------------------------------------------

_TWO_PIPELINE_CONFIG_YAML = """\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
vector_collections:
  - name: contract_chunks
    description: Contract text chunks.
  - name: rule_chunks
    description: Rule text chunks.
pipelines:
  - name: contracts
    routing_description: Vendor contracts to review for compliance.
    match:
      metadata:
        doc_type: contract
    steps:
      - tool: chunk-embed-upsert
        collection: contract_chunks
        chunker:
          type: fixed
          chunk_size: 512
          overlap: 64
  - name: rules
    routing_description: Company policy and standards documents.
    match:
      metadata:
        doc_type: rules
    steps:
      - tool: chunk-embed-upsert
        collection: rule_chunks
        chunker:
          type: fixed
          chunk_size: 512
          overlap: 64
"""


class TestBuildAppMatchMetadataFields:
    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_match_key_added_to_metadata_fields(self, mock_build_embedder, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        cfg = AppConfig.from_yaml(_TWO_PIPELINE_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        app = await build_app(cfg, system=SystemResources(vector_store=sys_vs), app_status="initializing")
        # Both pipelines match on doc_type — both collections should include it in metadata_fields
        contract_vc = app._pipelines[0]._vector_by_name["contract_chunks"]
        rule_vc = app._pipelines[0]._vector_by_name["rule_chunks"]
        assert "doc_type" in contract_vc.schema.metadata_fields
        assert "doc_type" in rule_vc.schema.metadata_fields

    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_explicit_metadata_fields_preserved_alongside_match_key(self, mock_build_embedder, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        yaml_with_explicit = _TWO_PIPELINE_CONFIG_YAML.replace(
            "  - name: contract_chunks\n    description: Contract text chunks.",
            "  - name: contract_chunks\n    description: Contract text chunks.\n    metadata_fields: [vendor]",
        )
        cfg = AppConfig.from_yaml(yaml_with_explicit)
        sys_vs = FAISSVectorStore()
        app = await build_app(cfg, system=SystemResources(vector_store=sys_vs), app_status="initializing")
        contract_vc = app._pipelines[0]._vector_by_name["contract_chunks"]
        assert "vendor" in contract_vc.schema.metadata_fields
        assert "doc_type" in contract_vc.schema.metadata_fields

    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_no_match_block_leaves_metadata_fields_unchanged(self, mock_build_embedder, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()
        app = await build_app(
            cfg,
            system=SystemResources(structured_store=system_store, vector_store=sys_vs),
            app_status="initializing",
        )
        vc = app._pipelines[0]._vector_by_name["document_chunks"]
        assert vc.schema.metadata_fields == []
