"""Factory — builds live app instances from parsed AppConfig."""

from __future__ import annotations

import os
from typing import Any

from api.config import AppConfig, ChunkerConfig, DocumentStoreConfig, EmbeddingConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.stores.base import StructuredStoreBase, VectorCollectionSchema, VectorStoreBase
from cogbase.core.app import CogBaseApp
from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema
from cogbase.pipeline.extraction.llm import LLMExtractor
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    StructuredCollection,
    SummarizeCollection,
    VectorCollection,
)
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


def build_document_store(cfg: DocumentStoreConfig) -> Any:
    """Instantiate a document store from its config."""
    if cfg.type == "local":
        from cogbase.stores.document.local_fs import LocalFSDocumentStore
        return LocalFSDocumentStore(cfg.path)  # type: ignore[arg-type]
    if cfg.type == "s3":
        from cogbase.stores.document.s3 import S3DocumentStore
        return S3DocumentStore(bucket=cfg.bucket, prefix=cfg.prefix, region=cfg.region)  # type: ignore[arg-type]
    raise ValueError(f"Unknown document_store type: {cfg.type!r}")


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


def _build_chunker(cfg: ChunkerConfig) -> Any:
    if cfg.type == "fixed":
        from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
        return FixedSizeChunker(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    if cfg.type == "langchain":
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(
                "langchain-text-splitters required: pip install langchain-text-splitters"
            ) from exc
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
) -> Any:
    """Instantiate a CogBase application from *config*.

    Iterates over all ``config.pipeline.steps`` to build each referenced
    collection.  Supports ``chunk-embed-upsert``, ``extract-structured``, and
    ``summarize-embed-upsert`` steps.

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
    smc_by_name = {smc.name: smc for smc in config.summarize_collections}

    # --- Shared resources (built once, referenced by multiple collections) ---
    vector_store: VectorStoreBase | None = None
    embedder = None

    needs_vector = any(
        s.tool in ("chunk-embed-upsert", "summarize-embed-upsert") for s in steps
    )
    if needs_vector:
        vector_store_cfg = config.vector_store or system_vector_store_cfg
        if vector_store_cfg is None:
            raise ValueError(
                "chunk-embed-upsert / summarize-embed-upsert steps require a vector store "
                "(configure vector_store in the app config or system config)"
            )
        if config.embedding is None:
            raise ValueError(
                "chunk-embed-upsert / summarize-embed-upsert steps require an embedding config"
            )
        vector_store = _build_vector_store(vector_store_cfg)
        embedder = _build_embedder(config.embedding, llm)

    structured_store: StructuredStoreBase | None = None
    needs_structured = any(s.tool == "extract-structured" for s in steps)
    if needs_structured:
        if config.structured_store is not None:
            structured_store = build_structured_store(config.structured_store)
        elif system_structured_store is not None:
            structured_store = system_structured_store
        else:
            raise ValueError(
                "extract-structured steps require a structured store "
                "(configure structured_store in the app config or system config)"
            )

    # --- Build collection objects (deduplicated per name) ---
    vector_collections: list[VectorCollection] = []
    structured_collections: list[StructuredCollection] = []
    summarize_collections: list[SummarizeCollection] = []
    built_vc: set[str] = set()
    built_sc: set[str] = set()
    built_smc: set[str] = set()

    for step in steps:
        if step.tool == "chunk-embed-upsert" and step.collection not in built_vc:
            vc_cfg = vc_by_name[step.collection]
            chunker = _build_chunker(vc_cfg.chunker)
            vector_collections.append(VectorCollection(
                schema=VectorCollectionSchema(
                    name=vc_cfg.name,
                    dimensions=vector_store_cfg.dim,  # type: ignore[union-attr]
                    description=vc_cfg.description,
                ),
                store=vector_store,  # type: ignore[arg-type]  # validated above
                embedder=embedder,   # type: ignore[arg-type]
                chunker=chunker,
            ))
            built_vc.add(vc_cfg.name)

        elif step.tool == "extract-structured" and step.collection not in built_sc:
            sc_cfg = sc_by_name[step.collection]
            extraction_model = build_model_from_json_schema(
                sc_cfg.schema_, model_name=sc_cfg.name.upper()
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
            structured_collections.append(StructuredCollection(
                schema=extractor.schema,  # LLMExtractor derives the schema from the model
                store=structured_store,   # type: ignore[arg-type]  # validated above
                extractor=extractor,
            ))
            built_sc.add(sc_cfg.name)

        elif step.tool == "summarize-embed-upsert" and step.collection not in built_smc:
            smc_cfg = smc_by_name[step.collection]
            summarize_collections.append(SummarizeCollection(
                schema=VectorCollectionSchema(
                    name=smc_cfg.name,
                    dimensions=vector_store_cfg.dim,  # type: ignore[union-attr]
                    description=smc_cfg.description,
                ),
                store=vector_store,    # type: ignore[arg-type]  # validated above
                embedder=embedder,     # type: ignore[arg-type]
                llm=llm,
                prompt=smc_cfg.prompt or "Summarize this document in a few sentences.",
                max_tokens=smc_cfg.max_tokens,
            ))
            built_smc.add(smc_cfg.name)

    pipeline = IngestionPipeline(
        name=config.name,
        steps=[(s.tool, s.collection) for s in steps],
        vector_collections=vector_collections or None,
        structured_collections=structured_collections or None,
        summarize_collections=summarize_collections or None,
    )

    document_store = build_document_store(config.document_store) if config.document_store else None
    return CogBaseApp(config.name, llm, pipeline, document_store=document_store)
