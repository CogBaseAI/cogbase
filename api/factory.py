"""Factory — builds live app instances from parsed AppConfig."""

from __future__ import annotations

from typing import Any

from cogbase.config.config import AppConfig, ChunkerConfig, ExtractorConfig
from cogbase.config.stores import StructuredStoreConfig
from cogbase.embeddings import build_embedding as _build_embedder
from cogbase.llms import build_llm as _build_llm
from cogbase.stores import (
    CollectionSchema,
    DocumentStoreBase,
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
from cogbase.core.basemodel_to_schema import cls_generate_schema
from cogbase.pipeline.extraction.llm import LLMExtractor
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    StructuredCollection,
    VectorCollection,
    PipelineStep,
)
from cogbase.workflows.runner import WorkflowRunner
from api.system_resources import SystemResources

import logging
logger = logging.getLogger(__name__)

DEFAULT_DOC_PROMPT = (
    "Summarize this document in a concise way, focusing on the most important "
    "points and avoiding unnecessary detail."
)

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
    system: SystemResources | None = None,
    app_status: str,
) -> Any:
    """Instantiate a CogBase application from *config*.

    Resources are resolved in priority order:

    1. Values declared explicitly in *config* — full per-application isolation.
    2. System-level resources supplied via *system* — shared across applications.
    3. No fallback — raises ``ValueError`` when a required resource is absent.
    """
    sys = system or SystemResources()

    # --- Top-level resources (independent of pipeline) ---
    llm = _build_llm(config.llm) if config.llm else sys.llm
    if llm is None:
        raise ValueError(
            "llm is required: set it in the app config or in the system config"
        )

    embedder = _build_embedder(config.embedding) if config.embedding else sys.embedder

    vector_store: VectorStoreBase | None = (
        _build_vector_store(config.vector_store) if config.vector_store else sys.vector_store
    )

    structured_store: StructuredStoreBase | None = (
        _build_structured_store(config.structured_store) if config.structured_store else sys.structured_store
    )

    document_store = (
        _build_document_store(config.document_store) if config.document_store else sys.document_store
    )

    # --- Vector collections ---
    vector_collections: list[VectorCollection] = []
    for vc_cfg in config.vector_collections:
        if vector_store is None:
            raise ValueError(
                f"vector collection {vc_cfg.name!r} requires a vector store"
                " (configure vector_store in the app config or system config)"
            )
        if embedder is None:
            raise ValueError(
                f"vector collection {vc_cfg.name!r} requires an embedding config"
            )
        vector_collections.append(VectorCollection(
            schema=VectorCollectionSchema(
                name=vc_cfg.name,
                dimensions=vc_cfg.dimensions,
                description=vc_cfg.description,
                metadata_fields=vc_cfg.metadata_fields,
            ),
            store=vector_store,
            embedder=embedder,
        ))

    # --- Structured collections ---
    structured_collections: list[StructuredCollection] = []
    structured_schemas: list[CollectionSchema] = []
    extractors_by_col: dict[str, LLMExtractor] = {}
    step_by_col: dict[str, Any] = {s.collection: s for p in config.pipelines for s in p.steps}
    for sc_cfg in config.structured_collections:
        if structured_store is None:
            raise ValueError(
                f"structured collection {sc_cfg.name!r} requires a structured store"
                " (configure structured_store in the app config or system config)"
            )

        record_model = build_model_from_json_schema(sc_cfg.schema_, model_name=sc_cfg.name.upper() + "_RECORD")
        sc_schema = CollectionSchema(
            name=sc_cfg.name,
            description=sc_cfg.description,
            primary_fields=sc_cfg.primary_fields,
            fields=cls_generate_schema(record_model),
        )

        structured_collections.append(StructuredCollection(schema=sc_schema, store=structured_store))
        structured_schemas.append(sc_schema)

        step = step_by_col.get(sc_cfg.name)
        ext_cfg: ExtractorConfig | None = step.extractor if step else None
        if ext_cfg is not None:
            extraction_model = build_model_from_json_schema(
                ext_cfg.extraction_schema, model_name=sc_cfg.name.upper() + "_EXTRACTION"
            )
            extractor = LLMExtractor(
                llm,
                extraction_model=extraction_model,
                record_model=record_model,
                config=ext_cfg,
            )
            extractors_by_col[sc_cfg.name] = extractor

    # --- Create collections in backing stores (idempotent) ---
    # Restored active apps already have their collections, so avoid the noisy
    # idempotent create calls and logs during startup reloads.
    if app_status != "active":
        for vc in vector_collections:
            await vc.store.create_collection(vc.schema)
            logger.info("created vector collection=%s, app=%s", vc.schema, config.name)
        for sc_schema in structured_schemas:
            await structured_store.create_collection(sc_schema)
            logger.info("created structured collection=%s, app=%s", sc_schema, config.name)

    # --- Pipelines ---
    pipelines: list[IngestionPipeline] = []
    for p_cfg in config.pipelines:
        pipeline_steps: list[PipelineStep] = []
        for s in p_cfg.steps:
            ps = PipelineStep(
                tool=s.tool,
                collection=s.collection,
                when=s.when.metadata if s.when else None,
            )
            if s.tool == "chunk-embed-upsert":
                chunker_cfg = s.chunker or ChunkerConfig()
                ps.chunker = _build_chunker(chunker_cfg)
            elif s.tool == "extract-structured":
                ps.extractor = extractors_by_col.get(s.collection)
            elif s.tool == "document-embed-upsert":
                ps.llm = llm
                ps.doc_prompt = s.doc_prompt or DEFAULT_DOC_PROMPT
            pipeline_steps.append(ps)
        pipelines.append(IngestionPipeline(
            name=p_cfg.name or config.name,
            steps=pipeline_steps,
            vector_collections=vector_collections or None,
            structured_collections=structured_collections or None,
            match=p_cfg.match.metadata if p_cfg.match else None,
            parallel=p_cfg.parallel,
        ))

    vc_schemas = [vc.schema for vc in vector_collections]

    qrunner = QueryRunner(
        llm=llm,
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        vector_schemas=vc_schemas or None,
        structured_schemas=structured_schemas or None,
        document_store=document_store,
        app_name=config.name,
    )

    # --- Workflows ---
    workflow_runners: dict[str, WorkflowRunner] = {}
    for wf_cfg in config.workflows:
        workflow_runners[wf_cfg.name] = WorkflowRunner(
            wf_cfg,
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            llm=llm,
        )
        logger.info("registered workflow=%s app=%s trigger=%s", wf_cfg.name, config.name, wf_cfg.trigger.type)

    return CogBaseApp(config.name, pipelines, qrunner, document_store=document_store, workflow_runners=workflow_runners)
