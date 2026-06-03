# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

CogBase is in active early development. The knowledge pipeline, workflow engine, query runner, skills registry, REST API, store adapters, app generator, document registry, and background task tracking are implemented. The short-term memory tier (session-local working context with token-budgeted assembly and compaction) is implemented in `cogbase/memory/`; the episodic and long-term tiers and the adaptive evolution engine are planned but not yet implemented.

## Architecture

CogBase is a framework for building AI applications that need to understand, cross-reference, and reason over large volumes of documents. The five key components are: a knowledge pipeline (ingest + structured extraction), composable workflows, a skill registry, a multi-tier memory system, and an adaptive evolution engine. It has six layers with clean boundaries:

**App Generator** (conversational)
- User describes document types, facts that matter, and example questions in natural language
- LLM generates a complete draft `config.yaml`: pipeline steps, vector/structured collections, extraction schemas, prompts, and workflows
- Draft is revised conversationally then deployed via `POST /generate/{session_id}/deploy`

**Knowledge Pipeline** (async, ingest-time)
- An app may have multiple named pipelines; documents are routed to a pipeline by metadata (e.g. `doc_type`)
- Three step types run in declaration order per document:
  - `chunk-embed-upsert` — splits text into overlapping passages, embeds, upserts to a vector collection
  - `extract-structured` — LLM extraction → typed records → structured collection
  - `document-embed-upsert` — LLM summary of the full document → embed → upsert as a single chunk per document to a vector collection
- Writes to pluggable structured and vector stores

**Workflows** (on-demand / after_ingest)
- YAML-declared sequential pipelines over already-ingested collections
- Four built-in tools: `structured-query`, `vector-search`, `llm-structured`, `structured-save`; support `foreach` loops
- Step parameters are Jinja2 templates with `input`, `steps.<id>`, and `item` namespaces
- Manual triggers use `params_from_collection`: accept `doc_id` and auto-derive params the same way `after_ingest` does
- Can be triggered manually via API or automatically after a successful ingest (`trigger.type: after_ingest`)
- Results streamed as SSE; sit between the pipeline (document-time) and skills (query-time)

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

**Adaptive Evolution** (background, planned)
- Gap detector mines episodic logs for signals the current config doesn't cover: low vector scores, repeated null answers, recurring tool chains
- Surfaces concrete suggestions (new field, new step, new skill) with supporting evidence via `GET /applications/{name}/suggestions`
- On user acceptance: config is patched and only affected documents are re-ingested

## Current project structure

```
cogbase/
├── cogbase/
│   ├── pipeline/             # chunking/, extraction/, ingestion_pipeline.py
│   ├── stores/               # base.py, schema.py, filters.py, structured/, vector/, document/
│   ├── skills/               # skill.py, registry.py
│   ├── embeddings/           # base.py, openai.py, huggingface.py
│   ├── llms/                 # base.py, openai.py
│   ├── tools/                # builtin/ (chunk_embed_upsert, extract)
│   ├── workflows/            # runner.py, context.py, tools/ (structured-query, vector-search, llm-structured, structured-save)
│   └── core/                 # app.py, app_generator.py, runner.py, session.py, models.py
├── api/                      # FastAPI REST API
│   ├── routers/              # applications.py, skills.py, workflows.py
│   ├── config.py             # AppConfig (YAML schema)
│   ├── factory.py            # build_app — registers schemas, wires pipelines
│   └── example_system_config.yaml
├── examples/
│   ├── contract_analyst_demo/
│   ├── contract_compliance_demo/
│   ├── legal_case_prep_demo/
│   └── vc_portfolio_demo/
└── docker-compose.yml
```

## Key interfaces

**Store adapters** — implement these to add a new backend:
```python
class StructuredStoreBase:
    async def create_collection(self, schema: CollectionSchema) -> None: ...
    async def save(self, collection: str, records: list[dict]) -> None: ...
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
        ("chunk-embed-upsert",    "document_chunks"),
        ("extract-structured",    "contracts"),
        ("document-embed-upsert", "document_summary"),
    ],
    chunk_collections=[ChunkCollection(schema=VectorCollectionSchema(name="document_chunks", ...), ...)],
    structured_collections=[StructuredCollection(schema=..., extractor=...)],
    document_collections=[DocumentCollection(schema=VectorCollectionSchema(name="document_summary", ...), ...)],
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

## REST API

Applications are created and managed through `POST /applications` (ZIP bundle upload). Key endpoints:

App generator:
- `POST /generate` — start a generation session from a natural-language description
- `POST /generate/{session_id}/revise` — revise the draft conversationally
- `POST /generate/{session_id}/deploy` — deploy the draft as a new application

Application lifecycle:
- `POST /applications` — create from ZIP bundle (config.yaml + referenced files)
- `POST /applications/{name}/upload_documents` — upload documents (saved to doc store; ingestion task handles the rest)
- `GET /applications/{name}/documents` — list all documents with workflow status
- `POST /applications/{name}/query` — blocking query
- `POST /applications/{name}/query/stream` — streaming query (SSE)
- `GET/POST/DELETE /applications/{name}/skills` — manage skills per application

Workflows:
- `POST /applications/{name}/workflows/{workflow_name}/stream` — run a workflow (SSE)

System:
- `POST /system/config` — configure LLM and embedding providers at runtime (no restart required)

Adaptive evolution (planned):
- `GET /applications/{name}/suggestions` — list pending suggestions with supporting evidence
- `POST /applications/{name}/suggestions/{id}/accept` — accept; triggers config patch + targeted re-ingest
- `POST /applications/{name}/suggestions/{id}/reject` — reject
