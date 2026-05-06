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
from cogbase.core.basemodel_to_schema import cls_generate_schema, cls_json_schema_for_llm
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
            ),
            store=vector_store,
            embedder=embedder,
            metadata_fields=vc_cfg.metadata_fields,
        ))

    # --- Structured collections ---
    structured_collections: list[StructuredCollection] = []
    structured_schemas: list[CollectionSchema] = []
    extractors_by_col: dict[str, LLMExtractor] = {}
    step_by_col: dict[str, Any] = {s.collection: s for s in (config.pipeline.steps if config.pipeline else [])}
    for sc_cfg in config.structured_collections:
        if structured_store is None:
            raise ValueError(
                f"structured collection {sc_cfg.name!r} requires a structured store"
                " (configure structured_store in the app config or system config)"
            )
        extraction_model = build_model_from_json_schema(
            sc_cfg.schema_, model_name=sc_cfg.name.upper()
        )
        step = step_by_col.get(sc_cfg.name)
        ext_cfg: ExtractorConfig | None = step.extractor if step else None
        if ext_cfg is not None:
            extract_as_list = ext_cfg.extract_as_list
            list_field = ext_cfg.list_field
            item_id_field = ext_cfg.item_id_field
            system_prompt = None
            if ext_cfg.prompt:
                if extract_as_list:
                    system_prompt = (
                        ext_cfg.prompt
                        + f'\nReturn a JSON object with a single key "{list_field}" whose value is an array.\n'
                        + "Each element must have these fields:\n\n"
                        + cls_json_schema_for_llm(extraction_model)
                    )
                else:
                    system_prompt = ext_cfg.prompt + cls_json_schema_for_llm(extraction_model)
            extractor = LLMExtractor(
                llm,
                extraction_model=extraction_model,
                collection_name=sc_cfg.name,
                collection_description=sc_cfg.description,
                extract_as_list=extract_as_list,
                list_field=list_field,
                item_id_field=item_id_field,
                system_prompt=system_prompt,
            )
            extractors_by_col[sc_cfg.name] = extractor
            sc_schema = extractor.schema
        else:
            sc_schema = CollectionSchema(
                name=sc_cfg.name,
                primary_fields=sc_cfg.primary_fields,
                fields=cls_generate_schema(extraction_model),
                description=sc_cfg.description,
            )
        structured_collections.append(StructuredCollection(schema=sc_schema, store=structured_store))
        structured_schemas.append(sc_schema)

    # --- Create collections in backing stores (idempotent) ---
    for vc in vector_collections:
        await vc.store.create_collection(vc.schema)
        logger.info("created vector collection=%s, app=%s", vc.schema, config.name)
    for sc_schema in structured_schemas:
        await structured_store.create_collection(sc_schema)
        logger.info("created structured collection=%s, app=%s", sc_schema, config.name)

    # --- Pipeline steps ---
    raw_steps = config.pipeline.steps if config.pipeline else []
    pipeline_steps: list[PipelineStep] = []
    for s in raw_steps:
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
            ps.prompt = s.prompt or "Summarize this document in a few sentences."
            ps.max_tokens = s.max_tokens
        pipeline_steps.append(ps)

    pipeline = IngestionPipeline(
        name=config.name,
        steps=pipeline_steps,
        vector_collections=vector_collections or None,
        structured_collections=structured_collections or None,
    )

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

    return CogBaseApp(config.name, pipeline, qrunner, document_store=document_store, workflow_runners=workflow_runners)
