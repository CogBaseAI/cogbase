# Changelog

## v0.3.0 — 2026-06-02

### Query Runner

- LLM token usage (`input_tokens`, `output_tokens`) is now counted across all LLM calls in a query and returned in `QueryResult` and `QueryResponse` (both blocking and streaming)
- Vector search deduplicates results across multiple calls — chunks already returned are excluded from subsequent `vector_search` tool calls in the same query
- Chunks from the same document are now sorted by character offset before being presented to the LLM, giving it coherent sequential context
- Citation-based filtering: only document slices actually cited by the LLM are included in the response; previously all fetched slices were returned
- Fixed a bug where cited chunks and slices were returning all collected results instead of only the ones the LLM referenced in its citations
- `structured_lookup` tool is only registered when the app has at least one structured collection — avoids a confusing no-op tool in vector-only apps
- Custom `system_prompt` can now be set at query time via `QueryRequest.system_prompt`, overriding the app-level default for a single request
- App-level `system_prompt` can be configured in `config.yaml` and takes effect for all queries to that app
- `top_k` is now settable in `QueryRequest` for per-request result tuning
- Improved `read_document` tool description: explicit guidance on how to read context before or after a retrieved chunk using `char_offset`

### Store Adapters

- `AppScope` scoping applied to all store adapters (structured, vector, document) — collections are now namespaced by app name, preventing conflicts between apps sharing the same backend
- Full cleanup on app deletion: vector and structured collections are dropped, system store records are removed, and all documents in the document store are deleted
- Allow hyphens (`-`) in collection names; resource name validation unified to a single function

### Knowledge Pipeline

- Langchain chunker updated to use configurable separators so splits occur at sentence boundaries rather than mid-word or mid-sentence
- Chinese sentence separator (`。`) added to the langchain chunker, enabling correct chunking of Chinese-language documents

### Demo UI

- New web UI with tabbed layout: Apps, Build, Ingest, Query, Demos, and Data tabs
- Integrated into the demo Docker image
- Query responses now include `document_slices` alongside chunks and structured records
- Unit tests for UI server

### Benchmarks

- GraphRAG benchmark: CogBase tested against novel and medical QA datasets with GPT-4o-mini; full results documented
- LoCoMo benchmark: CogBase scores 92.8% on the LoCoMo conversational memory benchmark, vs. Mem0's 91.6%; input/output token counts tracked per query

---

## v0.2.0 — 2026-05-25

### App Generator

- Split config generation into two focused LLM steps — pipeline config first, then workflow config — making the pipeline extraction schema the authoritative source for downstream workflow schemas
- Consolidated to a single `propose_app_config` tool, removing unnecessary round-trips to the LLM
- Auto-derive structured collection `schema` and `primary_fields` from the pipeline extractor's `record_mode` and `id_field`; no longer duplicated in the system prompt
- Hardened `AppConfig` validation: workflow collection references, no multi-extractor use of the same structured collection, `primary_fields` must be a subset of `output_schema`, no silent skip of invalid extraction JSON
- Strip `id_field` from LLM-generated extraction schemas (was incorrectly included), and handle the `record_mode.one` case where the LLM returns a single-item list
- Improved prompts: explicit `RecordMode` guidance, full workflow config example, double-quoted description strings to avoid YAML parse errors from embedded colons
- Fixed `_resolve_base_model_variants` not recursing into nested unions in the schema renderer
- Moved generator implementation to `cogbase/core/app_generator.py`; router layer is now thin
- Documented schema relationships among pipeline extraction schema, structured collection schema, and workflow schema at the top of `config.py`
- Added end-to-end app generation tests against `contract_analyst_demo` and `contract_compliance_demo` documents with real LLM calls

### Document Registry

- New per-app document registry tracks every ingested document (path, status, timestamps)
- `DocWorkflowRecord` is created at ingest time for any document that needs to pass through a workflow, giving a simple doc-level workflow view
- `DocWorkflowStatus` values: `PENDING`, `READY` (manual trigger, no task yet), `RUNNING`, `DONE`, `FAILED`
- `GET /applications/{name}/documents` lists all docs with their workflow status; surfaced in the demo client and UI

### Background Task Tracking

- Ingest and workflow runs are now tracked as `TaskRecord` entries in the task store
- `TaskRecord` carries `created_at` for scheduling metrics
- Ingestion tasks are self-contained and idempotent — a task can be re-run at any time without side effects
- Upload flow updated: document is saved to the document store first; the ingestion task then owns the rest of the pipeline
- `stream_workflow` checks pending tasks before starting so duplicate runs are avoided; removed the synchronous `run_workflow` API (clients hit timeout on long workflows)
- Demo UI updated to use the tasks API to show workflow pending status

### Workflow Engine

- Replaced `input_schema` with `params_from_collection` — manual triggers now accept `doc_id` and auto-derive params the same way `after_ingest` does
- Removed `WorkflowConfig.input_schema` from config and all example files

### LLM & Embedding Configuration

- LLM and embedding providers can now be configured at runtime via `POST /system/config` and persisted in the system database — no restart required
- Config is written back to the system YAML on startup so it survives image rebuilds
- Simplified config: `api_key` is required directly in `config.yaml`; `api_key_env` indirection removed
- Support for OpenAI-compatible LLM and embedding providers (any base URL)
- Demo UI: Settings tab for configuring providers; warning shown if LLM or embedding is not yet configured

### Knowledge Pipeline

- Introduced `ChunkerBase`: centralises `chunk_id` generation and `doc.metadata` inheritance, removes the redundant `chunk_index` metadata field
- Added `chunk_codec.py` — shared encode/decode/projection path used by all vector store backends
- `DocumentSlice` model added to `QueryResult` so `read_document` output is surfaced alongside passage chunks and structured records
- Fixed AUTO routing: LLM-routed documents now receive the same pipeline match metadata keys as metadata-routed documents
- Fixed pipeline routing: a pipeline with `match=None` no longer incorrectly captures documents that should be routed by the LLM
- Renamed `cogbase/pipeline/ingestion` → `cogbase/pipeline/chunking`

### Store Adapters

- SQLite: `create_collection` now detects and repairs NOT NULL / nullable constraint mismatches on schema evolution — no manual migration needed
- SQLite: fixed a binding error caused by nested Pydantic model fields being mapped to `STRING` instead of `JSON`
- Structured store: unified `save` API accepts `dict` only; Pydantic models are `model_dump`-ed at the call site
- Switched from a custom `build_model_from_json_schema` to the `jsonschema` library for schema validation and LLM prompting in `llm-structured` workflow steps
- `llm-structured` step retries on Pydantic validation errors; fixed `_unwrap_nullable` bug in `json_schema_to_basemodel`

### Demo UI

- Ingest tab shows per-document workflow status and a "Run workflow" button for pending documents
- Deploy and ingest progress indicators added
- Contract compliance demo re-enabled: detects new documents not yet processed by workflow
- "Check compliance" action unified under the Demos tab; shows only pending docs

### Examples

- New `legal_case_demo` — ingests a full case bundle (pleadings, evidence, depositions) and extracts structured facts for case preparation

### Bug Fixes

- Fixed `build_app` missing `task_store` argument
- Fixed `stream_workflow` timeout: long-running workflows are now fully async with task tracking
- App generator: strip YAML code fences from LLM output before parsing
- App generator: prevent `doc_id` cross-contamination when multiple pipelines handle different document types

### Infrastructure

- FastAPI now reads the package version from `importlib.metadata` — no more hard-coded version string
- `pytest` config consolidated into `pyproject.toml`; live tests (real external services) gated behind a marker
- Demo Docker image: LLM and embedding configuration is no longer required at image build time

---

## v0.1.0 — 2026-05-14

Initial release. CogBase is a framework for building AI applications from a plain-language description, with structured fact extraction, grounded LLM reasoning, and adaptive self-improvement from usage.

### What's included

**Knowledge Pipeline**
- Three-step ingestion pipeline: `chunk-embed-upsert` (passage-level vector search), `extract-structured` (LLM → typed records), `document-embed-upsert` (document-level summary embeddings)
- Multiple named pipelines per app, with LLM-based routing (metadata, auto, or llm strategies)
- File upload endpoint with automatic Markdown conversion
- Configurable mini LLM model for extraction and summarization to reduce cost

**Query Runner**
- Unified LLM agent loop — no fixed routing patterns
- Built-in tools: `structured_lookup` (exact filtered queries), `vector_search` (semantic over any collection)
- Responses include referenced chunks used by the LLM
- Streaming query responses over SSE

**Workflow Engine**
- YAML-declared sequential workflows over ingested collections
- Built-in tools: `structured-query`, `vector-search`, `llm-structured`, `structured-save`
- Jinja2 templates across `input`, `steps.<id>`, and `item` namespaces; `foreach` loops
- Trigger on-demand via API or automatically after successful ingest (`after_ingest`)
- Streaming results over SSE

**App Generator** _(beta)_
- Conversational app generation from a plain-language description
- LLM-delegated extraction schema and full app config (`config.yaml`) generation
- Revise the draft conversationally; deploy via REST API

**Skills Registry**
- Custom skill interface aligned with the [AgentSkills specification](https://agentskills.io/specification)
- Register and manage skills per application via REST API

**Store Adapters**
- Structured: in-memory (dev), SQLite, PostgreSQL
- Vector: in-memory + FAISS, pgvector
- Pluggable interface — implement two async classes to add a new backend

**REST API** (FastAPI)
- Full CRUD for applications, skills, workflows
- ZIP bundle deploy for reproducible app packaging
- Streaming endpoints for queries and workflows

**Example Applications**
- `contract_analyst_demo` — extract and query clauses from legal contracts
- `contract_compliance_demo` — compliance workflow triggered after ingest
- `vc_portfolio_demo` — structured extraction from VC deal memos

**Demo UI**
- Web UI bundled in the demo Docker image
- Shows extraction schema, query results with referenced chunks, and manual workflow triggers

### Not yet implemented
- Memory layer (short-term Redis, episodic, long-term)
- Adaptive Evolution engine (gap detection, suggestions, targeted re-ingest)

### Getting started

```bash
docker compose up
```

See [`docs/`](docs/) for API reference, architecture overview, and concepts.
