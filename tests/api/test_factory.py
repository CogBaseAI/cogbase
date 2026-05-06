"""Unit tests for api/factory.py — build_structured_store and build_app."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.config.config import AppConfig
from cogbase.config.stores import StructuredStoreConfig
from api.factory import build_app
from api.system_resources import SystemResources
from cogbase.stores import build_structured_store
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
pipeline:
  steps:
    - tool: extract-structured
      collection: contract_extraction
      extractor:
        type: llm
        extraction_schema: '{_EXTRACTION_SCHEMA}'
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
pipeline:
  parallel: true
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
        structured_store = next(iter(app._ingest_pipeline._structured_by_name.values())).store
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
        structured_store = next(iter(app._ingest_pipeline._structured_by_name.values())).store
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
        assert app._ingest_pipeline._vector_by_name == {}

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

        assert app._ingest_pipeline._vector_by_name

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

        assert "document_chunks" in app._ingest_pipeline._vector_by_name


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
pipeline:
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
pipeline:
  parallel: false
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

        assert "document_summary" in app._ingest_pipeline._vector_by_name

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

        vc = app._ingest_pipeline._vector_by_name["document_summary"]
        assert vc.name == "document_summary"
        step = next(s for s in app._ingest_pipeline._steps if s.collection == "document_summary")
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

        stores = {vc.store for vc in app._ingest_pipeline._vector_by_name.values()}
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

        assert app._ingest_pipeline._vector_by_name
        assert app._ingest_pipeline._structured_by_name
        assert "document_summary" in app._ingest_pipeline._vector_by_name

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

        assert "document_chunks" in app._ingest_pipeline._vector_by_name
        assert "document_summary" in app._ingest_pipeline._vector_by_name

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
pipeline:
  steps:
    - tool: extract-structured
      collection: contract_clauses
      extractor:
        type: llm
        extraction_schema: '{_LIST_EXTRACTION_SCHEMA}'
        record_mode: many
        response_field: clauses
        id_field: clause_id
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
pipeline:
  steps:
    - tool: extract-structured
      collection: contract_clauses
      extractor:
        type: llm
        extraction_schema: '{_LIST_EXTRACTION_SCHEMA}'
        record_mode: many
        response_field: clauses
        id_field: clause_id
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
pipeline:
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
        return next(s for s in app._ingest_pipeline._steps if s.collection == collection).extractor

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
