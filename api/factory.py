"""Factory — builds live app instances from parsed AppConfig."""

from __future__ import annotations

from typing import Any

from cogbase.config.config import AppConfig, ChunkerConfig
from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.embeddings import build_embedding as _build_embedder
from cogbase.llms import build_llm as _build_llm
from cogbase.stores import (
    StructuredStoreBase,
    VectorCollectionSchema,
    VectorStoreBase,
    build_document_store as _build_document_store,
    build_structured_store as _build_structured_store,
    build_vector_store as _build_vector_store,
)
from cogbase.core.app import CogBaseApp
from cogbase.core.query_runner import QueryRunner
from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema
from cogbase.pipeline.extraction.llm import LLMExtractor
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    StructuredCollection,
    DocumentCollection,
    ChunkCollection,
)
from cogbase.core.basemodel_to_schema import cls_json_schema_for_llm

import logging
logger = logging.getLogger(__name__)


def build_document_store(cfg: DocumentStoreConfig) -> Any:
    """Instantiate a document store from its config."""
    return _build_document_store(cfg)


def build_structured_store(cfg: StructuredStoreConfig) -> Any:
    """Instantiate a structured store from its config."""
    return _build_structured_store(cfg)


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


async def build_app(
    config: AppConfig,
    *,
    system_structured_store: StructuredStoreBase | None = None,
    system_vector_store_cfg: VectorStoreConfig | None = None,
    system_document_store_cfg: DocumentStoreConfig | None = None,
) -> Any:
    """Instantiate a CogBase application from *config*.

    Store backends are resolved in priority order:

    1. Values declared explicitly in *config* (``structured_store``,
       ``vector_store``, ``document_store``) — full per-application isolation.
    2. System-level stores supplied via the keyword arguments — the structured
       store is shared; collection names scope records to their collection.
    3. No fallback — raises ``ValueError`` when a declared collection has no
       backing store.
    """
    # --- Top-level resources (independent of pipeline) ---
    llm = _build_llm(config.llm)
    embedder = _build_embedder(config.embedding) if config.embedding else None

    vector_store_cfg = config.vector_store or system_vector_store_cfg
    vector_store: VectorStoreBase | None = (
        _build_vector_store(vector_store_cfg) if vector_store_cfg else None
    )

    if config.structured_store is not None:
        structured_store: StructuredStoreBase | None = build_structured_store(config.structured_store)
    elif system_structured_store is not None:
        structured_store = system_structured_store
    else:
        structured_store = None

    document_store_cfg = config.document_store or system_document_store_cfg
    document_store = build_document_store(document_store_cfg) if document_store_cfg else None

    # --- Collections (built from their own config, independent of pipeline steps) ---
    chunk_collections: list[ChunkCollection] = []
    for vc_cfg in config.chunk_collections:
        if vector_store is None:
            raise ValueError(
                f"chunk collection {vc_cfg.name!r} requires a vector store"
                " (configure vector_store in the app config or system config)"
            )
        if embedder is None:
            raise ValueError(
                f"chunk collection {vc_cfg.name!r} requires an embedding config"
            )
        chunk_collections.append(ChunkCollection(
            schema=VectorCollectionSchema(
                name=vc_cfg.name,
                dimensions=vc_cfg.dimensions,
                description=vc_cfg.description,
            ),
            store=vector_store,
            embedder=embedder,
            chunker=_build_chunker(vc_cfg.chunker),
        ))

    structured_collections: list[StructuredCollection] = []
    for sc_cfg in config.structured_collections:
        if structured_store is None:
            raise ValueError(
                f"structured collection {sc_cfg.name!r} requires a structured store"
                " (configure structured_store in the app config or system config)"
            )
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
            collection_description=sc_cfg.description,
            system_prompt=system_prompt,
        )
        structured_collections.append(StructuredCollection(
            schema=extractor.schema,
            store=structured_store,
            extractor=extractor,
        ))

    document_collections: list[DocumentCollection] = []
    for dc_cfg in config.document_collections:
        if vector_store is None:
            raise ValueError(
                f"document collection {dc_cfg.name!r} requires a vector store"
                " (configure vector_store in the app config or system config)"
            )
        if embedder is None:
            raise ValueError(
                f"document collection {dc_cfg.name!r} requires an embedding config"
            )
        document_collections.append(DocumentCollection(
            schema=VectorCollectionSchema(
                name=dc_cfg.name,
                dimensions=dc_cfg.dimensions,
                description=dc_cfg.description,
            ),
            store=vector_store,
            embedder=embedder,
            llm=llm,
            prompt=dc_cfg.prompt or "Summarize this document in a few sentences.",
            max_tokens=dc_cfg.max_tokens,
            metadata_fields=dc_cfg.metadata_fields,
        ))

    # --- Create collections in their backing stores (idempotent) ---
    for cc in chunk_collections:
        await cc.store.create_collection(cc.schema)
        logger.info("created chunk collection=%s, app=%s", cc.schema, config.name)
    for sc in structured_collections:
        await sc.store.create_collection(sc.schema)
        logger.info("created structured collection=%s, app=%s", sc.schema, config.name)
    for dc in document_collections:
        await dc.store.create_collection(dc.schema)
        logger.info("created document collection=%s, app=%s", dc.schema, config.name)

    # --- Pipeline (references already-built collections) ---
    steps = config.pipeline.steps if config.pipeline else []
    pipeline = IngestionPipeline(
        name=config.name,
        steps=[(s.tool, s.collection) for s in steps],
        chunk_collections=chunk_collections or None,
        structured_collections=structured_collections or None,
        document_collections=document_collections or None,
    )

    vc_schemas = [c.schema for c in [*chunk_collections, *document_collections]]

    runner = QueryRunner(
        llm=llm,
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        vector_schemas=vc_schemas or None,
        structured_schemas=[sc.schema for sc in structured_collections] or None,
        document_store=document_store,
        app_name=config.name,
    )

    return CogBaseApp(config.name, pipeline, runner, document_store=document_store)
