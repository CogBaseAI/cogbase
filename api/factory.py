"""Factory — builds live app instances from parsed AppConfig."""

from __future__ import annotations

import os
from typing import Any

from api.config import AppConfig, ChunkerConfig, EmbeddingConfig, StructuredStoreConfig, VectorStoreConfig


def _build_llm_client(config: AppConfig) -> Any:
    if config.llm.provider == "openai":
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        api_key = config.llm.resolved_api_key()
        return openai.AsyncOpenAI(api_key=api_key)
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider!r}")


def _build_structured_store(cfg: StructuredStoreConfig) -> Any:
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


def build_app(config: AppConfig) -> Any:
    """Instantiate a pack application from *config*.

    The returned object has ``setup()``, ``ingest()``, ``ingest_many()``, and
    ``query()`` methods but is not yet set up — call ``await app.setup()``
    before use.
    """
    llm_client = _build_llm_client(config)

    structured_store = _build_structured_store(config.structured_store)

    vector_store = None
    embedder = None
    chunker = None
    if config.vector_store is not None:
        vector_store = _build_vector_store(config.vector_store)
    if config.embedding is not None:
        embedder = _build_embedder(config.embedding, llm_client)
    if config.chunker is not None:
        chunker = _build_chunker(config.chunker)

    pack_name = config.pack.name if config.pack else "legal.contract_analyst"

    if pack_name == "legal.contract_analyst":
        from packs.legal.contract_analyst import LegalContractApp
        return LegalContractApp(
            client=llm_client,
            model=config.llm.model,
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            chunker=chunker,
        )

    raise ValueError(f"Unknown pack: {pack_name!r}")
