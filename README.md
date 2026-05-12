# CogBase

**Generate. Ingest. Reason. Evolve.**

Build AI apps from a plain-language description. Ingest anything. Extract structured facts. Reason across all of it. Get smarter with every query.

CogBase is an open-source framework for building AI applications that need to understand, cross-reference, and reason over large volumes of unstructured data — documents, emails, transcripts, chat logs, reports, and more.

It provides the foundational layer that vertical AI products are built on: typed fact extraction, a pluggable hybrid store, an LLM agent query runner, composable workflows and skills, a multi-tier memory system, and an adaptive evolution engine — generated from a plain-language description, deployed through a REST API, and improved by every query.

---

## The problem

Most RAG pipelines retrieve text and pass it to an LLM. That works for simple Q&A. It breaks down when you need to:

- Spot contradictions between two sources ("the contract says 60 days; the email says 30")
- Build a reliable timeline across dozens of documents
- Answer questions that require reasoning over structured facts, not just semantic similarity
- Ground generated output in citable, auditable sources
- Automate multi-step workflows across a large document set
- Maintain continuity across sessions and accumulate knowledge over time

CogBase solves this with a structured extraction layer sitting between raw ingestion and the LLM — turning unstructured input into typed, queryable facts before any reasoning begins — and a query runner that lets the LLM decide how to use structured and vector retrieval tools to answer questions.

---

## Architecture

CogBase is organized into six layers with clean boundaries between them.

```
╔═══════════════════════════════════════════════════════════╗
║  APP GENERATOR                 (conversational)           ║
║                                                           ║
║  User describes:                                          ║
║    • document types  ("SaaS contracts, vendor emails")    ║
║    • facts that matter  ("parties, payment terms, dates") ║
║    • example questions  ("which vendors auto-renew?")     ║
║          ↓                                                ║
║  LLM generates complete draft config:                     ║
║    vector collections  (passages + summaries)             ║
║    structured collections + extraction schemas + prompts  ║
║    pipeline steps  (chunk, extract, summarise)            ║
║    workflows  (if example questions need fan-out)         ║
║          ↓                                                ║
║  Draft presented → user revises conversationally          ║
║          ↓                                                ║
║  Deploy via POST /applications                            ║
╚═══════════════════════════════════════════════════════════╝
          ↓  config.yaml ZIP bundle
╔═══════════════════════════════════════════════════════════╗
║  KNOWLEDGE PIPELINE                        (async)        ║
║                                                           ║
║  Raw inputs                                               ║
║  (PDF, DOCX, email, chat, transcript, ...)                ║
║          ↓                                                ║
║  Ingestion & parsing                                      ║
║          ↓                                                ║
║  Steps run in order:                                      ║
║    chunk-embed-upsert     → passage chunks + embeddings   ║
║    extract-structured     → typed records via LLM         ║
║    document-embed-upsert  → one vector/doc like summary   ║
║          ↓                ↓                ↓              ║
║  ┌──────────────────┐  ┌──────────────────────────────┐   ║
║  │ Structured Store │  │       Vector Store           │   ║
║  │ (typed records,  │  │ document_chunks (passages)   │   ║
║  │  schemas, facts) │  │ document_summary (per-doc)   │   ║
║  └──────────────────┘  └──────────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════╝
          ↕  structured-query / vector-search / structured-save
╔═══════════════════════════════════════════════════════════╗
║  WORKFLOWS                         (on-demand)            ║
║                                                           ║
║  API call  ──or──  after_ingest trigger                   ║
║          ↓                                                ║
║  Sequential steps over ingested collections:              ║
║    structured-query   → read typed records                ║
║    vector-search      → semantic retrieval                ║
║    llm-structured     → LLM judge / classifier            ║
║    structured-save    → write derived records + SSE       ║
║          ↓                                                ║
║  Derived records land in output_collections               ║
╚═══════════════════════════════════════════════════════════╝
          ↕  hybrid retrieval tools
╔═══════════════════════════════════════════════════════════╗
║  QUERY RUNNER                              (real-time)    ║
║                                                           ║
║  User query                                               ║
║          ↓                                                ║
║  LLM agent loop                                           ║
║    ├── structured_lookup tool  (exact records)            ║
║    ├── vector_search tool      (passages or summaries)    ║
║    └── skill tools             (custom capabilities)      ║
║          ↓                                                ║
║  Grounded, cited response                                 ║
╚═══════════════════════════════════════════════════════════╝
          ↕  reads/writes
╔═══════════════════════════════════════════════════════════╗
║  MEMORY LAYER                              (persistent)   ║
║                                                           ║
║  Short-term  →  Redis / in-memory                         ║
║               (session-scoped context window)             ║
║                                                           ║
║  Episodic    →  Structured Store                          ║
║               (conversation + agent action history)       ║
║                                                           ║
║  Long-term   →  Structured Store + Vector Store           ║
║               (cross-session facts, conclusions,          ║
║                confirmed resolutions, user preferences)   ║
╚═══════════════════════════════════════════════════════════╝
          ↕  reads episodic history
╔═══════════════════════════════════════════════════════════╗
║  ADAPTIVE EVOLUTION                     (background)      ║
║                                                           ║
║  Gap detector mines episodic logs:                        ║
║    • low vector scores     → missing collection or step   ║
║    • repeated null answers → missing structured field     ║
║    • recurring tool chains → candidate skill              ║
║          ↓                                                ║
║  Suggestion queue: user confirms, adjusts, or rejects     ║
║          ↓  on acceptance                                 ║
║  Config patched → targeted re-ingest → app updated        ║
║          ↺  feeds back to App Generator                   ║
╚═══════════════════════════════════════════════════════════╝
```

See [docs/architecture.md](docs/architecture.md) for a detailed walkthrough of each layer.

---

## Project structure

```
cogbase/
├── cogbase/
│   ├── pipeline/     # ingestion, chunking, LLM extraction
│   ├── stores/       # structured + vector store adapters
│   ├── skills/       # skill interface + registry
│   ├── workflows/    # workflow engine
│   └── core/         # CogBaseApp, Runner, Session
├── api/              # FastAPI REST API + config schema
└── examples/         # demo applications (contract, compliance, VC)
```

---

## Core Concepts

- **App generator** — describe your documents and questions in plain language; the system generates the full `config.yaml`
- **Knowledge pipeline** — chunk-embed, extract structured facts, and summarize at ingest time; pluggable store backends
- **Workflows** — YAML-declared analytical pipelines with `foreach` loops and `after_ingest` triggers
- **Query runner** — LLM agent loop with `structured_lookup`, `vector_search`, and skill tools; no fixed routing pattern
- **Memory** — short-term (session), episodic (history), and long-term (cross-session) tiers
- **Adaptive evolution** — gap detector mines usage logs to surface concrete config improvement suggestions
- **Skills** — discrete, stateless custom capabilities registered per application

See [docs/concepts.md](docs/concepts.md) for details on each capability.

---

## Quickstart

The demo setup uses SQLite + FAISS — no external databases required. You need Docker and an OpenAI API key.

```bash
git clone https://github.com/cogbase/cogbase
cd cogbase/server
export OPENAI_API_KEY=sk-...
docker compose -f docker-compose.demo.yml up --build
```

The API is available at `http://localhost:8000`. API docs are at `http://localhost:8000/docs`.

See [`server/README.md`](server/README.md) for data persistence details and how to reset to a clean state.

### Create an application

```bash
# Build the bundle
zip my_app.zip config.yaml extraction_schema.json extraction_prompt.txt

# Create the application
curl -X POST http://localhost:8000/applications \
  -F bundle=@my_app.zip
```

### Ingest documents

```bash
curl -X POST http://localhost:8000/applications/my-contract-analyzer/ingest_documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"doc_id": "vendor-001", "text": "..."},
      {"doc_id": "nda-002",    "text": "..."}
    ],
    "concurrency": 5
  }'
```

### Query

```bash
curl -X POST http://localhost:8000/applications/my-contract-analyzer/query \
  -H "Content-Type: application/json" \
  -d '{"text": "which contracts expire before 2026?"}'
```

Or stream the response:

```bash
curl -N http://localhost:8000/applications/my-contract-analyzer/query/stream \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "summarise all termination rights across the vendor portfolio"}'
# data: {"token": "Here"}
# data: {"token": " are"}
# ...
# data: {"result": {"answer": "...", "passthrough": false, "structured_records": [...]}}
# data: [DONE]
```

---

## Examples

Domain-specific applications are provided as examples showing how to combine pipeline steps, schemas, and extractors for a particular vertical. They are intended as starting points — copy, adapt, and deploy via the REST API.

```
examples/
└── contract_analyst_demo/      # Legal contract extraction + Q&A
    ├── schema.py               # ContractRecord schema definition
    ├── saas_contracts.py       # SaaS contract demo data
    ├── demo.py                 # End-to-end ingestion and query demo
    └── README.md               # Extracted fields, query patterns, customisation
```

---

## Use cases

CogBase is not limited to legal. The core architecture maps to any domain where professionals spend significant time reading, cross-referencing, and drafting from large heterogeneous data sets.

| Vertical | Input data | Core value |
|---|---|---|
| Legal | Contracts, emails, depositions, filings | Contradiction detection, timeline, draft motions |
| Insurance claims | Medical records, police reports, policy docs | Coverage gap detection, settlement drafting |
| M&A due diligence | Contracts, financials, IP filings, HR records | Risk surfacing, diligence memo generation |
| Financial compliance | Transaction records, policies, communications | Policy violation detection, audit reports |
| Medical records review | EHR notes, lab results, imaging reports, referrals | Drug conflict detection, care summary drafting |
| Academic / patent research | Papers, patents, citations | Prior art timelines, claim contradiction analysis |

About 90% of the codebase — the ingestion pipeline, workflow engine, query runner, skill registry, memory layer, and store interfaces — is identical across all verticals. You write the config and schema once. The store adapters handle the rest.

---

## REST API

See [docs/api.md](docs/api.md) for the full endpoint reference.

For working config examples, see:
- [`examples/contract_analyst_demo/config.yaml`](../examples/contract_analyst_demo/config.yaml)
- [`examples/contract_compliance_demo/config.yaml`](../examples/contract_compliance_demo/config.yaml)
- [`examples/vc_portfolio_demo/config.yaml`](../examples/vc_portfolio_demo/config.yaml)
---

## Roadmap

**Implemented (v0.1)**
- [x] Core ingestion pipeline (chunk-embed-upsert, extract-structured, document-embed-upsert)
- [x] Typed fact extraction with configurable JSON schema
- [x] Store adapter interfaces (StructuredStoreBase, VectorStoreBase)
- [x] Built-in adapters: SQLite, Postgres, FAISS, pgvector
- [x] Per-document summarization vector collection
- [x] LLM agent query loop with structured_lookup, vector_search, and read_document tools
- [x] Skill registry + base skill interface
- [x] REST API (create/update/delete apps, ingest, query, streaming)
- [x] Declarative workflow engine (structured-query, vector-search, llm-structured, structured-save; foreach; after_ingest)
- [x] Docker Compose quickstart (SQLite + FAISS, see `server/`)
- [x] Contract analyst, contract compliance, and VC portfolio examples

**Improvements**

These are known gaps in the first-pass implementations.

- [ ] Native document parsing — pipeline currently requires plain text; add PDF, DOCX, and HTML ingestion, etc
- [ ] Query runner: auto-compaction — `compact_messages` is implemented but not wired into the loop, etc
- [ ] Workflow step timeouts and partial-failure recovery — a failing step currently aborts the whole workflow, parallel steps, etc
- [ ] API layer - authentication (API keys or token-based), etc
- [ ] Broader integration test coverage — especially for query runner loops, workflows, and API end-to-end paths

**Planned**
- [ ] App generator (conversational config generation from description + example questions, with iterative revision)
- [ ] Short-term memory (Redis + in-memory)
- [ ] Episodic memory (conversation + agent history)
- [ ] Long-term memory (cross-session knowledge store)
- [ ] Adaptive evolution engine (gap detector: retrieval score analysis, null-answer mining, tool-chain clustering)
- [ ] Suggestion surface API (GET /suggestions, accept/reject with targeted re-ingest)
- [ ] Insurance example
- [ ] Medical records example
- [ ] Managed cloud hosting (SOC 2)

---

## Contributing

CogBase is in early development. The highest-impact contributions right now are improvements to the existing implementations — not new features.

**Harden existing work** (see _Improvements in the roadmap for specifics)
- Fix known gaps in the pipeline, query runner, workflows, and API
- Add integration tests for the query runner loop, workflow execution, and API end-to-end paths
- Try the [quickstart](#quickstart) with real documents and file issues for anything that breaks or behaves unexpectedly

**Extend the framework**
- **Contribute a store adapter** — implement `StructuredStoreBase` or `VectorStoreBase` for a backend not yet supported
- **Contribute an example** — a YAML config + JSON schema + prompt file for a new vertical
- **Contribute a skill** — any stateless capability that implements the skill interface

**Build planned features**
- **Memory layer** — short-term (Redis), episodic (history), and long-term (cross-session facts) tiers
- **Adaptive evolution** — gap detector, suggestion queue, and targeted re-ingest
- **App generator** — conversational config generation from a plain-language description

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0
