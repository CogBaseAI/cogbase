# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

CogBase is in active early development. The knowledge pipeline, query runner, skills registry, REST API, and store adapters are implemented. The memory layer (short-term, episodic, long-term) and contradiction detection engine are planned but not yet implemented.

## Architecture

CogBase is a framework for structured fact extraction, contradiction detection, and grounded LLM reasoning over large document sets. It has three layers with clean boundaries:

**Knowledge Pipeline** (async, ingest-time)
- Three step types run in declaration order per document:
  - `chunk-embed-upsert` — splits text into overlapping passages, embeds, upserts to a vector collection
  - `extract-structured` — LLM extraction → typed records → structured collection
  - `summarize-embed-upsert` — LLM summary of the full document → embed → upsert as a single chunk per document to a vector collection
- Writes to pluggable structured and vector stores

**Query Runner** (real-time, query-time)
- Unified LLM agent loop — no fixed routing patterns
- LLM decides which tools to call based on the query:
  - `structured_lookup` — exact record queries with field filters
  - `vector_search` — semantic search against any named vector collection (passage chunks or document summaries)
  - skill tools — custom capabilities registered with the application
- Passthrough rule: large structured result sets are returned directly without LLM synthesis
- `Runner` handles both retrieval-only and skill-routing modes

**Memory Layer** (persistent, planned)
- Short-term: Redis-backed session context
- Episodic: conversation + agent action history in structured store
- Long-term: cross-session confirmed facts, resolved contradictions, preferences

## Current project structure

```
cogbase/
├── cogbase/
│   ├── pipeline/         # ingestion/, extraction/, ingestion_pipeline.py
│   ├── stores/           # base.py, schema.py, filters.py, structured/, vector/
│   ├── skills/           # skill.py, registry.py
│   ├── embeddings/       # base.py, openai.py, huggingface.py
│   ├── llms/             # base.py, openai.py
│   ├── tools/            # builtin/ (chunk_embed_upsert, extract)
│   └── core/             # app.py, runner.py, session.py, models.py
├── api/                  # FastAPI REST API
│   ├── routers/          # applications.py, skills.py
│   ├── config.py         # AppConfig (YAML schema)
│   ├── factory.py        # build_app from config
│   └── example_config.yaml
├── examples/
│   └── contract_analyst_demo/
└── docker-compose.yml
```

## Key interfaces

**Store adapters** — implement these to add a new backend:
```python
class StructuredStoreBase:
    async def create_collection(self, schema: CollectionSchema) -> None: ...
    async def save(self, collection: str, records: list[BaseModel]) -> None: ...
    async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]: ...
    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

class VectorStoreBase:
    async def upsert(self, collection: str, chunks: list[Chunk]) -> None: ...
    async def search(self, collection: str, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]: ...
    async def delete(self, collection: str, doc_id: str) -> None: ...
```

All public/abstract methods are async. CPU-bound implementations use `run_in_executor`.

**IngestionPipeline** — wraps collections and steps:
```python
pipeline = IngestionPipeline(
    name="legal",
    steps=[
        ("chunk-embed-upsert",     "document_chunks"),
        ("extract-structured",     "contracts"),
        ("summarize-embed-upsert", "document_summary"),
    ],
    vector_collections=[ChunkCollection(schema=VectorCollectionSchema(name="document_chunks", ...), ...)],
    structured_collections=[StructuredCollection(schema=..., extractor=...)],
    summarize_collections=[SummarizeCollection(schema=VectorCollectionSchema(name="document_summary", ...), ...)],
)
```

**VectorCollectionSchema** carries: `name`, `dimensions`, `description` (shown to LLM in retrieval prompt), optional `metadata`.

**CollectionSchema** carries: `name`, `primary_fields`, `fields` (dict of FieldSchema), `description` (shown to LLM in retrieval prompt).

**Skill interface** — aligned with the [AgentSkills specification](https://agentskills.io/specification):
```python
class Skill:
    name: str           # required — max 64 chars, lowercase alphanumeric + hyphens
    description: str    # required — shown to LLM when selecting a skill; max 1024 chars
    compatibility: str  # optional — environment requirements
    metadata: dict      # optional — arbitrary str→str key-value pairs
    allowed_tools: list # optional — tools this skill may invoke
    def run(self, input: dict, session: Session) -> dict: ...
```

## Domain examples

Domain-specific applications are in `examples/`, not in a `packs/` directory. Each example shows how to configure the pipeline, schema, and extractor for a specific vertical. Apps are deployed via the REST API using a ZIP bundle containing `config.yaml` and referenced files.

## Contradiction detection approach (planned)

Two-phase (not a single LLM prompt over long context):
1. Extract typed facts from each source individually
2. Cross-document comparison using embedding distance + NLI classification, bucketed by conflict type (date, numeric, statement)

Previously resolved contradictions are stored in long-term memory and excluded from future scans.

## REST API

Applications are created and managed through `POST /applications` (ZIP bundle upload). Key endpoints:
- `POST /applications` — create from ZIP bundle (config.yaml + referenced files)
- `POST /applications/{name}/ingest_documents` — ingest a batch of documents
- `POST /applications/{name}/query` — blocking query
- `POST /applications/{name}/query/stream` — streaming query (SSE)
- `GET/POST/DELETE /applications/{name}/skills` — manage skills per application
