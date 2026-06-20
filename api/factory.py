"""Factory — builds live app instances from parsed AppConfig."""

from __future__ import annotations

import json as _json
from typing import Any

from cogbase.config.config import (
    AppConfig,
    ChunkerConfig,
    DocumentEmbedUpsertStepConfig,
    ExtractStructuredStepConfig,
    ExtractorConfig,
    ChunkEmbedUpsertStepConfig,
)
from cogbase.config.stores import StructuredStoreConfig
from cogbase.embeddings import build_embedding as _build_embedder
from cogbase.llms import build_llm as _build_llm
from cogbase.stores import (
    AppScope,
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
from cogbase.core.query_runner import MemoryTiers, QueryRunner, RetrievalResources
from cogbase.memory import Distiller, EpisodicMemory, LongTermMemory, ShortTermMemory
from cogbase.stores.schema import FieldSchema, FieldType
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


def _json_schema_to_collection_fields(schema: dict) -> dict[str, FieldSchema]:
    """Map JSON Schema properties to FieldSchema — replaces cls_generate_schema."""
    properties = schema.get("properties", {})
    result: dict[str, FieldSchema] = {}
    for field_name, field_schema in properties.items():
        # unwrap anyOf nullable: [{"type": "X"}, {"type": "null"}]
        any_of = field_schema.get("anyOf")
        if any_of:
            non_null = [s for s in any_of if s.get("type") != "null"]
            field_schema = non_null[0] if non_null else {"type": "string"}
        t = field_schema.get("type")
        if t == "integer":
            result[field_name] = FieldSchema(type=FieldType.INTEGER)
        elif t == "number":
            result[field_name] = FieldSchema(type=FieldType.FLOAT)
        elif t == "boolean":
            result[field_name] = FieldSchema(type=FieldType.BOOLEAN)
        elif t in ("object", "array") or "$ref" in field_schema:
            result[field_name] = FieldSchema(type=FieldType.JSON)
        else:
            result[field_name] = FieldSchema(type=FieldType.STRING)
    return result


def _build_chunker(cfg: ChunkerConfig) -> Any:
    if cfg.type == "fixed":
        from cogbase.pipeline.chunking.fixed import FixedSizeChunker
        return FixedSizeChunker(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    if cfg.type == "langchain":
        from cogbase.pipeline.chunking.langchain import build_recursive_chunker
        return build_recursive_chunker(cfg.chunk_size, cfg.overlap)
    raise ValueError(f"Unknown chunker type: {cfg.type!r}")


async def build_app(
    config: AppConfig,
    *,
    app_id: str,
    system: SystemResources | None = None,
    app_status: str,
    task_store: Any | None = None,
) -> Any:
    """Instantiate a CogBase application from *config*.

    *app_id* is the application's stable internal id (distinct from the mutable
    client-facing ``config.name``); it drives the store scope prefix and the
    per-app document-store collection, so storage survives a rename.

    Resources are resolved in priority order:

    1. Values declared explicitly in *config* — full per-application isolation.
    2. System-level resources supplied via *system* — shared across applications.
    3. No fallback — raises ``ValueError`` when a required resource is absent.
    """
    sys = system or SystemResources()
    app_scope = AppScope(app_id=app_id)

    # --- Top-level resources (independent of pipeline) ---
    llm = _build_llm(config.llm) if config.llm else sys.llm
    if llm is None:
        raise ValueError(
            "llm is required: set it in the app config or in the system config"
        )

    embedder = _build_embedder(config.embedding) if config.embedding else sys.embedder

    vector_store: VectorStoreBase | None = (
        _build_vector_store(config.vector_store, scope=app_scope)
        if config.vector_store
        else (sys.vector_store.with_scope(app_scope) if sys.vector_store else None)
    )

    structured_store: StructuredStoreBase | None = (
        _build_structured_store(config.structured_store, scope=app_scope)
        if config.structured_store
        else (sys.structured_store.with_scope(app_scope) if sys.structured_store else None)
    )

    document_store = (
        _build_document_store(config.document_store, scope=app_scope)
        if config.document_store
        else (sys.document_store.with_scope(app_scope) if sys.document_store else None)
    )

    # --- Vector collections ---
    # Collect routing match keys per collection so they're always stored on chunks,
    # regardless of whether the config author remembered to list them in metadata_fields.
    match_keys_by_collection: dict[str, set[str]] = {}
    for p_cfg in config.pipelines:
        if p_cfg.match:
            for step in p_cfg.steps:
                if step.tool in ("chunk-embed-upsert", "document-embed-upsert"):
                    match_keys_by_collection.setdefault(step.collection, set()).update(
                        p_cfg.match.metadata.keys()
                    )

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
        extra = match_keys_by_collection.get(vc_cfg.name, set())
        metadata_fields = list(dict.fromkeys(vc_cfg.metadata_fields + list(extra)))
        vector_collections.append(VectorCollection(
            schema=VectorCollectionSchema(
                name=vc_cfg.name,
                dimensions=vc_cfg.dimensions,
                description=vc_cfg.description,
                metadata_fields=metadata_fields,
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

        record_schema = _json.loads(sc_cfg.schema_)
        sc_schema = CollectionSchema(
            name=sc_cfg.name,
            description=sc_cfg.description,
            primary_fields=sc_cfg.primary_fields,
            fields=_json_schema_to_collection_fields(record_schema),
        )

        structured_collections.append(StructuredCollection(schema=sc_schema, store=structured_store))
        structured_schemas.append(sc_schema)

        step = step_by_col.get(sc_cfg.name)
        ext_cfg: ExtractorConfig | None = step.extractor if isinstance(step, ExtractStructuredStepConfig) else None
        if ext_cfg is not None:
            extraction_schema = _json.loads(ext_cfg.extraction_schema)
            extractor = LLMExtractor(
                llm,
                extraction_schema=extraction_schema,
                config=ext_cfg,
                record_schema=record_schema,
                app_id=app_id,
            )
            extractors_by_col[sc_cfg.name] = extractor

    # --- Create collections in backing stores (idempotent) ---
    if app_status != "active":
        for vc in vector_collections:
            await vc.store.create_collection(vc.schema)
            logger.info("created vector collection=%s, app=%s", vc.schema.name, config.name)
        for sc_schema in structured_schemas:
            await structured_store.create_collection(sc_schema)
            logger.info("created structured collection=%s, app=%s", sc_schema.name, config.name)
    else:
        # Active apps already have their tables — skip DDL but populate the
        # in-memory schema registry so save/query can look up column types.
        for sc_schema in structured_schemas:
            structured_store.register_schema(sc_schema)

    # --- Pipelines ---
    pipelines: list[IngestionPipeline] = []
    for p_cfg in config.pipelines:
        pipeline_steps: list[PipelineStep] = []
        for s in p_cfg.steps:
            ps = PipelineStep(
                tool=s.tool,
                collection=s.collection,
            )
            if isinstance(s, ChunkEmbedUpsertStepConfig):
                ps.chunker = _build_chunker(s.chunker)
            elif isinstance(s, ExtractStructuredStepConfig):
                ps.extractor = extractors_by_col.get(s.collection)
            elif isinstance(s, DocumentEmbedUpsertStepConfig):
                ps.llm = llm
                ps.doc_prompt = s.doc_prompt or DEFAULT_DOC_PROMPT
            pipeline_steps.append(ps)
        pipelines.append(IngestionPipeline(
            name=p_cfg.name or config.name,
            description=p_cfg.routing_description,
            steps=pipeline_steps,
            vector_collections=vector_collections or None,
            structured_collections=structured_collections or None,
            match=p_cfg.match.metadata if p_cfg.match else None,
            parallel=p_cfg.parallel,
            app_id=app_id,
        ))
        logger.info(
            "registered pipeline=%s app=%s match=%s steps=%d",
            p_cfg.name,
            config.name,
            p_cfg.match,
            len(p_cfg.steps),
        )

    vc_schemas = [vc.schema for vc in vector_collections]

    # Episodic memory: the durable append-only event log.  Wired from the shared
    # (system) log store — deliberately unscoped, so events from every app land
    # in one log family and carry ``app_name`` for attribution (cross-app mining
    # by the future evolution engine).  Engaged only when a query carries a
    # session_id; absent a log store, the runner simply records nothing.
    episodic = EpisodicMemory(sys.log_store) if sys.log_store is not None else None

    # Short-term memory: session-local working context, projected from the same
    # episodic log (it has no store of its own).  Created only when episodic is
    # available, and shares that instance so the thread it assembles and the
    # events the runner records — including the session_compacted summaries it
    # appends during compaction — ride the same per-session stream and flush.
    short_term = (
        ShortTermMemory(episodic=episodic, llm=llm) if episodic is not None else None
    )

    # Long-term memory: curated cross-session knowledge.  Like the episodic log
    # it is system-level infrastructure — physically in the shared system stores,
    # logically partitioned per app by wrapping them in the app scope so each app
    # addresses its own prefixed collections (isolation by store layout, not a
    # query-time predicate; see docs/long-term-memory.md#the-app-is-the-partition-boundary).
    # Requires the system structured + vector stores and an embedder.
    long_term = None
    if (
        sys.structured_store is not None
        and sys.vector_store is not None
        and embedder is not None
    ):
        # Collections are created lazily on first use (the vector dimensionality
        # is learned from the first embedding), so build_app does no long-term
        # DDL — keeping app construction's collection-creation behaviour unchanged.
        long_term = LongTermMemory(
            sys.structured_store.with_scope(app_scope),
            sys.vector_store.with_scope(app_scope),
            llm,
            embedder,
            app_id=app_id,
            reconcile_guidance=config.memory.reconcile_guidance,
            recall_neighbors=config.memory.recall_neighbors,
        )

    # Distiller: offline promotion of durable records out of a settled session
    # log.  Needs both the log to read and the long-term store to write into.
    distiller = (
        Distiller(
            episodic,
            long_term,
            llm,
            domain_fact_guidance=config.memory.domain_fact_guidance,
            existing_memory_limit=config.memory.existing_memory_limit,
            single_call=config.memory.single_call_reconcile,
        )
        if episodic is not None and long_term is not None
        else None
    )

    # Resolve the skills assigned to this app (referenced by id) into loaded
    # Skill objects the runner can route to and execute.
    skills: list = []
    if config.skills and sys.skill_registry is not None:
        for skill_id in config.skills:
            try:
                skills.append(sys.skill_registry.get(skill_id))
            except KeyError:
                logger.warning("skill id=%s assigned to app=%s is not registered; skipping", skill_id, config.name)

    qrunner = QueryRunner(
        app_id=app_id,
        llm=llm,
        resources=RetrievalResources(
            document_store=document_store,
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            structured_schemas=structured_schemas or None,
            vector_schemas=vc_schemas or None,
        ),
        memory=MemoryTiers(short_term=short_term, episodic=episodic, long_term=long_term),
        skills=skills or None,
    )

    # --- Workflows ---
    workflow_runners: dict[str, WorkflowRunner] = {}
    for wf_cfg in config.workflows:
        workflow_runners[wf_cfg.name] = WorkflowRunner(
            wf_cfg,
            app_id=app_id,
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            llm=llm,
        )
        logger.info("registered workflow=%s app=%s trigger=%s", wf_cfg.name, config.name, wf_cfg.trigger.type)

    if document_store is None:
        raise ValueError(
            "document_store is required: set it in the app config or in the system config"
        )
    if structured_store is None:
        raise ValueError(
            "structured_store is required: set it in the app config or in the system config"
        )
    if task_store is None:
        raise ValueError(
            "task_store is required: provide it when calling build_app"
        )

    logger.info(
        "build app=%s, pipelines=%d routing=%s, workflows=%d vector_collections=%d structured_collections=%d",
        config.name,
        len(config.pipelines),
        config.pipeline_routing,
        len(config.workflows),
        len(config.vector_collections),
        len(config.structured_collections),
    )
    return CogBaseApp(
        config.name,
        pipelines,
        qrunner,
        app_id=app_id,
        document_store=document_store,
        structured_store=structured_store,
        workflow_runners=workflow_runners,
        llm=llm,
        routing_strategy=config.pipeline_routing.strategy,
        task_store=task_store,
        query_prompt=config.query_prompt,
        short_term=short_term,
        episodic=episodic,
        long_term=long_term,
        distiller=distiller,
    )
