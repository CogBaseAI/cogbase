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
from cogbase.core.runner import Runner
from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema
from cogbase.pipeline.extraction.llm import LLMExtractor
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    StructuredCollection,
    DocumentCollection,
    ChunkCollection,
)
from cogbase.core.basemodel_to_schema import cls_json_schema_for_llm


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


def build_app(
    config: AppConfig,
    *,
    system_structured_store: StructuredStoreBase | None = None,
    system_vector_store_cfg: VectorStoreConfig | None = None,
    system_document_store_cfg: DocumentStoreConfig | None = None,
) -> Any:
    """Instantiate a CogBase application from *config*.

    Iterates over all ``config.pipeline.steps`` to build each referenced
    collection.  Supports ``chunk-embed-upsert``, ``extract-structured``, and
    ``document-embed-upsert`` steps.

    Store backends are resolved in priority order:

    1. Values declared explicitly in *config* (``structured_store``,
       ``vector_store``, ``document_store``) — full per-application isolation.
    2. System-level stores supplied via the keyword arguments — the structured
       store is shared; collection names scope records to their collection.
    3. No fallback — raises ``ValueError`` when neither is provided.
    """
    llm = _build_llm(config.llm)

    steps = config.pipeline.steps if config.pipeline else []
    vc_by_name = {vc.name: vc for vc in config.chunk_collections}
    sc_by_name = {sc.name: sc for sc in config.structured_collections}
    dc_by_name = {dc.name: dc for dc in config.document_collections}

    # --- Shared resources (built once, referenced by multiple collections) ---
    vector_store: VectorStoreBase | None = None
    embedder = None

    needs_vector = any(
        s.tool in ("chunk-embed-upsert", "document-embed-upsert") for s in steps
    )
    if needs_vector:
        vector_store_cfg = config.vector_store or system_vector_store_cfg
        if vector_store_cfg is None:
            raise ValueError(
                "chunk-embed-upsert / document-embed-upsert steps require a vector store "
                "(configure vector_store in the app config or system config)"
            )
        if config.embedding is None:
            raise ValueError(
                "chunk-embed-upsert / document-embed-upsert steps require an embedding config"
            )
        vector_store = _build_vector_store(vector_store_cfg)
        embedder = _build_embedder(config.embedding)

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
    chunk_collections: list[ChunkCollection] = []
    structured_collections: list[StructuredCollection] = []
    document_collections: list[DocumentCollection] = []
    built_vc: set[str] = set()
    built_sc: set[str] = set()
    built_dc: set[str] = set()

    for step in steps:
        if step.tool == "chunk-embed-upsert" and step.collection not in built_vc:
            vc_cfg = vc_by_name[step.collection]
            chunker = _build_chunker(vc_cfg.chunker)
            chunk_collections.append(ChunkCollection(
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
                collection_description=sc_cfg.description,
                system_prompt=system_prompt,
            )
            structured_collections.append(StructuredCollection(
                schema=extractor.schema,  # LLMExtractor derives the schema from the model
                store=structured_store,   # type: ignore[arg-type]  # validated above
                extractor=extractor,
            ))
            built_sc.add(sc_cfg.name)

        elif step.tool == "document-embed-upsert" and step.collection not in built_dc:
            dc_cfg = dc_by_name[step.collection]
            document_collections.append(DocumentCollection(
                schema=VectorCollectionSchema(
                    name=dc_cfg.name,
                    dimensions=vector_store_cfg.dim,  # type: ignore[union-attr]
                    description=dc_cfg.description,
                ),
                store=vector_store,    # type: ignore[arg-type]  # validated above
                embedder=embedder,     # type: ignore[arg-type]
                llm=llm,
                prompt=dc_cfg.prompt or "Summarize this document in a few sentences.",
                max_tokens=dc_cfg.max_tokens,
                metadata_fields=dc_cfg.metadata_fields,
            ))
            built_dc.add(dc_cfg.name)

    pipeline = IngestionPipeline(
        name=config.name,
        steps=[(s.tool, s.collection) for s in steps],
        chunk_collections=chunk_collections or None,
        structured_collections=structured_collections or None,
        document_collections=document_collections or None,
    )

    # Determine default vector collection: first chunk-embed then document-embed, in step order.
    default_vc: str | None = None
    for step in steps:
        if step.tool == "chunk-embed-upsert":
            default_vc = step.collection
            break
    if default_vc is None:
        for step in steps:
            if step.tool == "document-embed-upsert":
                default_vc = step.collection
                break

    document_store_cfg = config.document_store or system_document_store_cfg
    document_store = build_document_store(document_store_cfg) if document_store_cfg else None

    _vc_schemas = [c.schema for c in [*chunk_collections, *document_collections]]

    runner = Runner(
        llm=llm,
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        default_vector_collection=default_vc,
        vector_schemas=_vc_schemas or None,
        structured_schemas=[sc.schema for sc in structured_collections] or None,
        document_store=document_store,
        app_name=config.name,
    )

    return CogBaseApp(config.name, pipeline, runner, document_store=document_store)
