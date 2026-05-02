"""Unit tests for api/factory.py — build_structured_store and build_app."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.config.config import AppConfig
from cogbase.config.stores import StructuredStoreConfig
from api.factory import build_app
from cogbase.stores import build_structured_store
from cogbase.pipeline.ingestion_pipeline import DocumentCollection, ChunkCollection
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

_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'

_EXTRACT_ONLY_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
structured_collections:
  - name: contract_extraction
    schema: '{_SCHEMA}'
    extractor:
      type: llm
pipeline:
  steps:
    - tool: extract-structured
      collection: contract_extraction
"""

_FULL_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
chunk_collections:
  - name: document_chunks
    chunker:
      type: fixed
      chunk_size: 512
      overlap: 64
structured_collections:
  - name: contract_extraction
    schema: '{_SCHEMA}'
    extractor:
      type: llm
pipeline:
  parallel: true
  steps:
    - tool: chunk-embed-upsert
      collection: document_chunks
    - tool: extract-structured
      collection: contract_extraction
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
            await build_app(cfg)

    @patch("api.factory._build_llm")
    async def test_uses_system_store_when_no_app_store(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_EXTRACT_ONLY_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(cfg, system_structured_store=system_store)
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
        app = await build_app(cfg, system_structured_store=system_store)
        structured_store = next(iter(app._ingest_pipeline._structured_by_name.values())).store
        assert isinstance(structured_store, SQLiteStructuredStore)


class TestBuildAppVectorStoreResolution:
    @patch("api.factory._build_llm")
    async def test_no_vector_collection_when_no_chunk_step(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_EXTRACT_ONLY_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = await build_app(cfg, system_structured_store=system_store)
        assert app._ingest_pipeline._chunk_by_name == {}

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
                system_structured_store=system_store,
                system_vector_store=sys_vs,
            )

        assert app._ingest_pipeline._chunk_by_name

    @patch("api.factory._build_llm")
    async def test_vector_collection_name_matches_config(self, mock_build_llm):
        """The vector collection name comes from chunk_collections config, not app name."""
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system_structured_store=system_store,
                system_vector_store=sys_vs,
            )

        assert "document_chunks" in app._ingest_pipeline._chunk_by_name


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
document_collections:
  - name: document_summary
    prompt: "Summarize in one sentence."
    max_tokens: 128
pipeline:
  steps:
    - tool: document-embed-upsert
      collection: document_summary
"""

_THREE_STEP_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
chunk_collections:
  - name: document_chunks
    chunker:
      type: fixed
      chunk_size: 512
      overlap: 64
structured_collections:
  - name: contract_extraction
    schema: '{_SCHEMA}'
    extractor:
      type: llm
document_collections:
  - name: document_summary
    prompt: "Summarize in one sentence."
    max_tokens: 128
pipeline:
  parallel: false
  steps:
    - tool: chunk-embed-upsert
      collection: document_chunks
    - tool: extract-structured
      collection: contract_extraction
    - tool: document-embed-upsert
      collection: document_summary
"""


class TestBuildAppDocumentCollection:
    @patch("api.factory._build_llm")
    async def test_document_collection_present_in_pipeline(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_SUMMARIZE_ONLY_CONFIG_YAML)
        sys_vs = FAISSVectorStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(cfg, system_vector_store=sys_vs)

        assert "document_summary" in app._ingest_pipeline._document_by_name

    @patch("api.factory._build_llm")
    async def test_document_collection_name_prompt_max_tokens(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_SUMMARIZE_ONLY_CONFIG_YAML)
        sys_vs = FAISSVectorStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(cfg, system_vector_store=sys_vs)

        dc = app._ingest_pipeline._document_by_name["document_summary"]
        assert dc.name == "document_summary"
        assert dc.prompt == "Summarize in one sentence."
        assert dc.max_tokens == 128

    @patch("api.factory._build_llm")
    async def test_document_collection_uses_shared_vector_store(self, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        cfg = AppConfig.from_yaml(_THREE_STEP_CONFIG_YAML)
        sys_vs = FAISSVectorStore()
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = await build_app(
                cfg,
                system_structured_store=system_store,
                system_vector_store=sys_vs,
            )

        vc = next(iter(app._ingest_pipeline._chunk_by_name.values()))
        dc = app._ingest_pipeline._document_by_name["document_summary"]
        assert vc.store is dc.store

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
                system_structured_store=system_store,
                system_vector_store=sys_vs,
            )

        assert app._ingest_pipeline._chunk_by_name
        assert app._ingest_pipeline._structured_by_name
        assert "document_summary" in app._ingest_pipeline._document_by_name

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
                system_structured_store=system_store,
                system_vector_store=sys_vs,
            )

        assert "document_chunks" in app._ingest_pipeline._chunk_by_name
        assert "document_summary" in app._ingest_pipeline._document_by_name

    @patch("api.factory._build_llm")
    @patch("api.factory._build_embedder")
    async def test_summarize_step_without_vector_store_raises(self, mock_build_embedder, mock_build_llm):
        mock_build_llm.return_value = _mock_llm()
        mock_build_embedder.return_value = MagicMock()
        cfg = AppConfig.from_yaml(_SUMMARIZE_ONLY_CONFIG_YAML)
        # No vector store supplied
        with pytest.raises(ValueError, match="vector store"):
            await build_app(cfg)


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
        app = await build_app(cfg, system_document_store=sys_doc)

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
        app = await build_app(cfg, system_document_store=sys_doc)

        assert isinstance(app.document_store, LocalFSDocumentStore)
        assert app.document_store._root.name == "app-docs"  # type: ignore[union-attr]
