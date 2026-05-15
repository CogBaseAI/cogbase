# Changelog

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
