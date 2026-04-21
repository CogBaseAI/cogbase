"""Factory — builds live app instances from parsed AppConfig."""

from __future__ import annotations

import os
from typing import Any

from api.config import AppConfig, ChunkerConfig, EmbeddingConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.stores.base import StructuredStoreBase


def _build_llm_client(config: AppConfig) -> Any:
    if config.llm.provider == "openai":
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        api_key = config.llm.resolved_api_key()
        return openai.AsyncOpenAI(api_key=api_key)
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider!r}")


def build_structured_store(cfg: StructuredStoreConfig) -> Any:
    """Instantiate a structured store from its config."""
    if cfg.type == "memory":
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        return InMemoryStructuredStore()
    if cfg.type == "sqlite":
        from cogbase.stores.structured.sqlite import SQLiteStructuredStore
        return SQLiteStructuredStore(cfg.path)  # path validated by config
    if cfg.type == "postgres":
        from cogbase.stores.structured.postgres import PostgresStructuredStore
        return PostgresStructuredStore(cfg.url)  # url validated by config
    raise ValueError(f"Unknown structured_store type: {cfg.type!r}")


def _build_vector_store(cfg: VectorStoreConfig) -> Any:
    if cfg.type == "faiss":
        from cogbase.stores.vector.faiss_store import FAISSVectorStore
        return FAISSVectorStore(dim=cfg.dim)
    if cfg.type == "pgvector":
        from cogbase.stores.vector.pgvector_store import PGVectorStore
        return PGVectorStore(dim=cfg.dim, dsn=cfg.url)
    raise ValueError(f"Unknown vector_store type: {cfg.type!r}")


def _build_embedder(cfg: EmbeddingConfig, llm_client: Any) -> Any:
    if cfg.provider == "openai":
        from cogbase.embeddings.openai import OpenAIEmbedding
        kwargs: dict[str, Any] = {}
        if cfg.dimensions is not None:
            kwargs["dimensions"] = cfg.dimensions
        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        return OpenAIEmbedding(client, model=cfg.model, **kwargs)
    if cfg.provider == "sentence-transformers":
        from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
        return SentenceTransformersEmbedding(model_name=cfg.model)
    raise ValueError(f"Unsupported embedding provider: {cfg.provider!r}")


def _build_chunker(cfg: ChunkerConfig) -> Any:
    if cfg.type == "fixed":
        from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
        return FixedSizeChunker(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    if cfg.type == "langchain":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from cogbase.pipeline.ingestion.langchain import LangChainChunker
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.overlap
        )
        return LangChainChunker(splitter)
    raise ValueError(f"Unknown chunker type: {cfg.type!r}")


def build_app(
    config: AppConfig,
    *,
    system_structured_store: StructuredStoreBase | None = None,
    system_vector_store_cfg: VectorStoreConfig | None = None,
    app_namespace: str | None = None,
) -> Any:
    """Instantiate a CogBase application from *config*.

    Store backends are resolved in priority order:

    1. Values declared explicitly in *config* (``structured_store``,
       ``vector_store``) — full per-application isolation.
    2. System-level stores supplied via the keyword arguments — the structured
       store is shared with other applications using collection-name namespacing;
       the vector store type is used to create a fresh per-application instance.
    3. Built-in fallback — an isolated in-memory structured store; no vector store.

    The returned object has ``setup()``, ``ingest_documents()``, and
    ``query()`` methods but is not yet set up — call ``await app.setup()``
    before use.

    Args:
        config:                   Parsed application config.
        system_structured_store:  Shared structured store from system config.
        system_vector_store_cfg:  Vector store type/settings from system config;
                                  a new instance is created per application.
        app_namespace:            Prefix applied to all collection names when
                                  using *system_structured_store*.  Defaults to
                                  ``config.name``.
    """
    llm_client = _build_llm_client(config)

    # --- Structured store ---------------------------------------------------
    # Priority: app config > system shared store (namespaced) > in-memory
    if config.structured_store is not None:
        structured_store = build_structured_store(config.structured_store)
    elif system_structured_store is not None:
        from api.namespaced_store import NamespacedStructuredStore
        ns = app_namespace or config.name
        structured_store: Any = NamespacedStructuredStore(system_structured_store, ns)
    else:
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        structured_store = InMemoryStructuredStore()

    # --- Vector store -------------------------------------------------------
    # Priority: app config > system config (new instance per app) > None
    vector_store_cfg = config.vector_store or (
        system_vector_store_cfg if config.embedding is not None else None
    )
    vector_store = _build_vector_store(vector_store_cfg) if vector_store_cfg else None

    embedder = _build_embedder(config.embedding, llm_client) if config.embedding else None
    chunker = _build_chunker(config.chunker) if config.chunker else None

    pack_name = config.pack.name if config.pack else "legal.contract_analyst"

    if pack_name == "legal.contract_analyst":
        from cogbase.core.app import CogBaseApp
        from cogbase.pipeline.extraction.llm import LLMExtractor
        from cogbase.stores.schema_util import cls_json_schema_for_llm
        from examples.contract_analyst_demo.schema import (
            CONTRACTS_COLLECTION,
            CONTRACTS_SYSTEM_PROMPT_PREFIX,
            ContractExtraction,
        )

        if config.extraction_schema is not None:
            from cogbase.core.json_schema import build_model_from_json_schema
            extraction_model = build_model_from_json_schema(
                config.extraction_schema, model_name="DynamicContractExtraction"
            )
            collection_name = CONTRACTS_COLLECTION
            id_field = "contract_id"
            system_prompt = CONTRACTS_SYSTEM_PROMPT_PREFIX + cls_json_schema_for_llm(extraction_model)
        else:
            extraction_model = ContractExtraction
            collection_name = CONTRACTS_COLLECTION
            id_field = "contract_id"
            system_prompt = CONTRACTS_SYSTEM_PROMPT_PREFIX + cls_json_schema_for_llm(ContractExtraction)

        extractor = LLMExtractor(
            llm_client,
            config.llm.model,
            extraction_model=extraction_model,
            collection_name=collection_name,
            id_field=id_field,
            system_prompt=system_prompt,
        )

        return CogBaseApp(
            client=llm_client,
            model=config.llm.model,
            extractors=[extractor],
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            chunker=chunker,
        )

    raise ValueError(f"Unknown pack: {pack_name!r}")
