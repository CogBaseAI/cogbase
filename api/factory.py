"""Factory — builds live app instances from parsed AppConfig."""

from __future__ import annotations

import os
from typing import Any

from api.config import AppConfig, EmbeddingConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.stores.base import StructuredStoreBase
from cogbase.core.app import CogBaseApp
from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema
from cogbase.pipeline.extraction.llm import LLMExtractor
from cogbase.core.basemodel_to_schema import cls_json_schema_for_llm
from cogbase.llms.base import LLMBase
from cogbase.llms.openai import OpenAILLM


def _build_llm(config: AppConfig) -> LLMBase:
    if config.llm.provider == "openai":
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        api_key = config.llm.resolved_api_key()
        client = openai.AsyncOpenAI(api_key=api_key)
        return OpenAILLM(client, model=config.llm.model)
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider!r}")


def build_structured_store(cfg: StructuredStoreConfig) -> Any:
    """Instantiate a structured store from its config."""
    if cfg.type == "memory":
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        return InMemoryStructuredStore()
    if cfg.type == "sqlite":
        from cogbase.stores.structured.sqlite import SQLiteStructuredStore
        return SQLiteStructuredStore(cfg.path)
    if cfg.type == "postgres":
        from cogbase.stores.structured.postgres import PostgresStructuredStore
        return PostgresStructuredStore(cfg.url)
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


def build_app(
    config: AppConfig,
    *,
    system_structured_store: StructuredStoreBase | None = None,
    system_vector_store_cfg: VectorStoreConfig | None = None,
) -> Any:
    """Instantiate a CogBase application from *config*.

    Iterates over ``config.pipeline.steps`` to determine which collections to
    activate.  Currently supports one ``chunk_and_embed`` step (vector) and one
    ``extract`` step (structured).

    Store backends are resolved in priority order:

    1. Values declared explicitly in *config* (``structured_store``,
       ``vector_store``) — full per-application isolation.
    2. System-level stores supplied via the keyword arguments — the structured
       store is shared; collection names scope records to their collection.
    3. No fallback — raises ``ValueError`` when neither is provided.
    """
    llm = _build_llm(config)

    steps = config.pipeline.steps if config.pipeline else []
    vc_by_name = {vc.name: vc for vc in config.vector_collections}
    sc_by_name = {sc.name: sc for sc in config.structured_collections}

    # --- Vector collection (chunk-embed-upsert step) -------------------------
    chunk_step = next((s for s in steps if s.tool == "chunk-embed-upsert"), None)
    vector_store = None
    embedder = None
    chunker = None
    vector_collection_name = None

    if chunk_step and chunk_step.collection in vc_by_name:
        vc_cfg = vc_by_name[chunk_step.collection]
        vector_collection_name = vc_cfg.name
        vector_store_cfg = config.vector_store or system_vector_store_cfg
        if vector_store_cfg is None:
            raise ValueError(
                f"chunk_and_embed step for '{vc_cfg.name}' requires a vector store "
                "(configure vector_store in the app config or system config)"
            )
        if config.embedding is None:
            raise ValueError(
                f"chunk_and_embed step for '{vc_cfg.name}' requires an embedding config"
            )
        vector_store = _build_vector_store(vector_store_cfg)
        embedder = _build_embedder(config.embedding, llm)
        chunker_cfg = vc_cfg.chunker
        if chunker_cfg.type == "fixed":
            from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
            chunker = FixedSizeChunker(chunk_size=chunker_cfg.chunk_size, overlap=chunker_cfg.overlap)
        elif chunker_cfg.type == "langchain":
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from cogbase.pipeline.ingestion.langchain import LangChainChunker
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunker_cfg.chunk_size, chunk_overlap=chunker_cfg.overlap
            )
            chunker = LangChainChunker(splitter)
        else:
            raise ValueError(f"Unknown chunker type: {chunker_cfg.type!r}")

    # --- Structured collection (extract-structured step) ---------------------
    extract_step = next((s for s in steps if s.tool == "extract-structured"), None)
    structured_store = None
    extractor = None

    if extract_step and extract_step.collection in sc_by_name:
        sc_cfg = sc_by_name[extract_step.collection]

        if config.structured_store is not None:
            structured_store = build_structured_store(config.structured_store)
        elif system_structured_store is not None:
            structured_store = system_structured_store
        else:
            raise ValueError(
                f"extract step for '{sc_cfg.name}' requires a structured store "
                "(configure structured_store in the app config or system config)"
            )

        extraction_model = build_model_from_json_schema(
            sc_cfg.schema_, model_name="DynamicExtraction"
        )

        system_prompt = None
        if sc_cfg.extractor.prompt:
            system_prompt = sc_cfg.extractor.prompt + cls_json_schema_for_llm(extraction_model)

        extractor = LLMExtractor(
            llm,
            extraction_model=extraction_model,
            collection_name=sc_cfg.name,
            system_prompt=system_prompt,
        )

    return CogBaseApp(
        name=config.name,
        llm=llm,
        extractor=extractor,
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        vector_collection_name=vector_collection_name,
    )
