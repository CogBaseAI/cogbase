# CogBase

**Generate. Ingest. Reason. Evolve.**

Build AI apps from a plain-language description. Ingest anything. Extract structured facts. Reason across all of it. Get smarter with every query.

CogBase is an open-source framework for building AI applications that need to understand, cross-reference, and reason over large volumes of unstructured data — documents, emails, transcripts, chat logs, reports, and more.

It provides the foundational layer that vertical AI products are built on: typed fact extraction, contradiction detection, a pluggable hybrid store, an LLM agent query runner, composable skills, a multi-tier memory system, and an adaptive evolution engine — generated from a plain-language description, deployed through a REST API, and improved by every query.

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

CogBase is organized into four layers with clean boundaries between them.

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

**App Generator** is the entry point for new applications. Instead of writing `config.yaml` by hand, describe your documents and example questions in natural language and the system generates the full configuration — collections, steps, schemas, prompts, and workflows — as a draft you can then revise conversationally before deploying.

**Knowledge Pipeline** runs asynchronously at ingest time. Three step types can be combined in any order, with optional `when_meta` predicates to route specific document types to different steps:

- `chunk-embed-upsert` — splits document text into overlapping passages, embeds them, and upserts into a vector collection for passage-level semantic search
- `extract-structured` — runs a configurable LLM extractor to produce typed records stored in a structured collection
- `document-embed-upsert` — generates one vector such as LLM summary per document, embeds it, and upserts into a vector collection for document-level semantic search

Both stores are pluggable — swap backends without changing application code.

**Workflows** run on-demand (via API call) or automatically after a successful ingest (`after_ingest` trigger). They are YAML-declared sequential pipelines over already-ingested collections — reading from structured and vector stores, calling an LLM to judge or classify, and writing derived records back to output collections. They stream results as SSE. Workflows sit between the pipeline (document-time) and skills (query-time, LLM-callable), handling analytical computations that need to fan out over many records but don't belong in the ingest step itself.

**Query Runner** drives a real-time LLM agent loop. Rather than a fixed routing pattern, the LLM receives the available tools and decides which to call: `structured_lookup` for exact record queries, `vector_search` against any configured vector collection (passage chunks, document summaries, or both), and any skill tools registered with the application. The loop continues until the LLM has enough evidence to produce a final answer. Large structured result sets are returned directly without synthesis (passthrough rule).

**Memory Layer** serves the layers above. Short-term memory holds the assembled context for the current query. Episodic memory logs the full history of queries, answers, and agent actions. Long-term memory accumulates confirmed facts, resolved contradictions, learned patterns, and user preferences across sessions.

**Adaptive Evolution** closes the feedback loop between usage and configuration. A background gap detector mines episodic logs for signals that the current config doesn't cover what users actually ask: low-scoring retrieval results suggest a missing collection or pipeline step; repeated "I don't have that information" answers suggest a missing structured field; recurring multi-step tool chains suggest a skill worth encapsulating. The system surfaces these as concrete, evidence-backed suggestions — a new extraction field, a new workflow, a new skill stub — and waits for user confirmation before applying any change. On acceptance, the config is patched and only the affected documents are re-ingested. The app evolves to fit how people actually use it.

---

## Core capabilities

### App generator

Instead of authoring `config.yaml` manually, describe what you want to build and let the system generate the full configuration:

1. **Describe your use case** — the document types you have (contracts, medical records, emails, transcripts), the facts that matter, and a handful of example questions you want to answer.
2. **Review the draft** — the system generates a complete `config.yaml` with pipeline steps, vector and structured collections, extraction schemas, extraction prompts, and any workflows needed to answer your example questions.
3. **Revise conversationally** — adjust any part of the generated config through follow-up chat: add a field, rename a collection, change a workflow step, tighten an extraction prompt.
4. **Deploy** — when satisfied, submit the config directly via `POST /generate/{session_id}/deploy` (equivalent to `POST /applications` with the generated bundle).

The generator is opinionated: it infers the minimal set of collections and steps needed to cover the example questions, defaulting to one passage-chunk vector collection, one document-summary vector collection, and one or more structured collections with typed fields. Workflows are only generated when example questions require multi-record fan-out (e.g., "flag all contracts that…").

**Example input:**

```json
{
  "description": "I review SaaS vendor contracts and need to track payment terms, renewal dates, and liability caps.",
  "document_type": "legal contracts (PDF / DOCX)",
  "example_questions": [
    "Which contracts expire before Q2 2026?",
    "Which vendors have auto-renewal clauses?",
    "What is the average liability cap across the portfolio?"
  ]
}
```

**Generated output (summarised):**

- Vector collections: `contract_chunks` (passage-level), `contract_summaries` (per-document)
- Structured collection: `contracts` — fields: `vendor`, `effective_date`, `expiry_date`, `payment_terms`, `auto_renewal` (bool), `liability_cap`, `doc_id`
- Pipeline steps: `chunk-embed-upsert → extract-structured → document-embed-upsert`
- Extraction prompt: tailored to pull the declared fields from contract text
- Workflow: `flag_auto_renewal` — queries `contracts` for `auto_renewal = true` and streams results (triggered by the third example question)

The response includes the full draft YAML, a human-readable summary of what was generated, and a `session_id` for follow-up revisions.

---

### Structured extraction

Every document is processed into structured records at ingestion time. Extraction is general — any JSON schema works: facts, entities, clauses, events, relationships, risk flags, and more. Each extractor declares the collection it writes to and its schema.

The built-in `Fact` model carries: `type`, `value`, `raw_text`, `doc_id`, `page`, `confidence`. The `raw_text` field is preserved verbatim from the source and used as the citation.

### Per-document summarization

Alongside passage chunks, the pipeline supports a `document-embed-upsert` step that generates one vector such as LLM summary per document and stores its embedding as a single vector. This gives the query runner two levels of semantic retrieval:

- **document_chunks** — precise, passage-level retrieval for detailed or specific questions
- **document_summary** — topic-level retrieval for high-level questions about what documents cover

The LLM automatically picks the right collection based on the query.

### Workflows

Workflows are named, YAML-declared analytical pipelines that run over already-ingested collections. They compose four built-in tools in any sequence, including `foreach` loops over result sets:

| Tool | What it does |
|---|---|
| `structured-query` | Read typed records with equality filters; result at `steps.<id>.records` |
| `vector-search` | Embed a query string and search a vector collection; result at `steps.<id>.chunks` |
| `llm-structured` | Call the LLM with a system prompt and JSON input, validate against a JSON Schema; result at `steps.<id>.output` |
| `structured-save` | Upsert records into a collection and stream each one to the caller; result at `steps.<id>.records` |

Step parameters are Jinja2 templates with three namespaces: `input` (invocation params), `steps.<id>` (prior step outputs), and `item` (current foreach element). A `{{ expr }}` that resolves to a list returns an actual Python list, not a string.

Workflows can be triggered manually via `POST /applications/{name}/workflows/{workflow_name}/run` or automatically after each successful document ingest (`trigger.type: after_ingest`, optionally gated by document metadata). Blocking and streaming (`/stream`) endpoints are both available.

### Contradiction detection

CogBase uses a two-phase approach rather than a single "find contradictions" prompt, which is unreliable over long context:

1. Extract typed facts from each source individually
2. Run a cross-document comparison pass over the fact store, using embedding distance + NLI classification to flag conflicts by type (date conflicts, numeric conflicts, statement conflicts)

This makes contradiction detection a query over structured data, not a needle-in-a-haystack prompt. Previously resolved contradictions are stored in long-term memory and not re-flagged in future sessions.

### LLM agent query loop

The query runner drives a multi-turn LLM agent loop with configurable retrieval tools:

| Tool | Description |
|---|---|
| `structured_lookup` | Exact record query against a named collection with field filters |
| `vector_search` | Semantic search against a named vector collection (chunks or summaries) |
| `skill tools` | Custom capabilities registered with the application |

The LLM calls tools as needed to gather evidence, then synthesises a grounded answer. No fixed routing pattern — the model decides. When `structured_lookup` returns a large result set (above the passthrough token threshold), records are returned directly as formatted text without an additional synthesis step.

### Pluggable stores

CogBase defines clean adapter interfaces for both stores. Swap backends via config — no application code changes required.

```python
from cogbase.stores import StructuredStoreBase, VectorStoreBase, CollectionSchema, VectorCollectionSchema, Filter
from cogbase.core.models import Chunk

class MyStructuredStore(StructuredStoreBase):
    async def create_collection(self, schema: CollectionSchema) -> None: ...
    async def save(self, collection: str, records: list[BaseModel]) -> None: ...
    async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]: ...
    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

class MyVectorStore(VectorStoreBase):
    async def upsert(self, collection: str, chunks: list[Chunk]) -> None: ...
    async def search(self, collection: str, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]: ...
    async def delete(self, collection: str, doc_id: str) -> None: ...
```

Built-in adapters: SQLite + FAISS (local/dev), Postgres + pgvector (production).

### Memory

CogBase maintains three tiers of memory, each scoped and persisted differently:

| Tier | Scope | Purpose |
|---|---|---|
| Short-term | Session | Assembled context window for the current query; expires with the session |
| Episodic | User / session | Full history of queries, answers, and agent actions; enables follow-ups and agent continuity |
| Long-term | User / project / org | Confirmed facts, resolved contradictions, learned patterns, preferences; persists indefinitely |

### Adaptive evolution

An initial app configuration is always a guess — the schema fields, pipeline steps, and skills defined up front reflect what the builder *expected* people to ask, not what they actually ask. Adaptive evolution lets the app correct itself over time.

A background gap detector continuously analyzes the episodic memory of a deployed application:

| Signal | What it indicates | Suggested action |
|---|---|---|
| Vector search scores are consistently low for a query class | The relevant information isn't being stored as vectors, or the chunking strategy is wrong | Add or reconfigure a collection or pipeline step |
| Queries return "I don't have that information" repeatedly | A fact users care about isn't being extracted | Add a field to the structured schema and re-extract |
| The same long tool chain appears repeatedly across sessions | A recurring multi-step reasoning pattern | Encapsulate as a named skill |
| An extraction field is never queried | The field isn't useful | Simplify the schema |

Each signal becomes a **suggestion** — a concrete, evidence-backed proposed change to the app's config. Suggestions are queued and surfaced via the API with supporting evidence (example queries, relevant session IDs, retrieval score distributions). The user reviews each suggestion, adjusts if needed, and accepts or rejects it.

On acceptance:
1. The app config is patched (a new field, a new step, a new skill stub)
2. A targeted re-ingest runs over the affected documents
3. The app's behavior updates without any manual config editing

This makes the app generator not just a one-time tool but a continuous improvement loop — each round of real usage makes the app more capable for the next.

### Skills

Skills are the unit of custom capability in CogBase — discrete, stateless, and composable. Each skill is a markdown file (describing what the LLM should do) alongside an optional Python implementation. Skills are loaded from a directory at server startup and can be assigned to applications via the REST API.

The skill interface is aligned with the [AgentSkills specification](https://agentskills.io/specification):

```python
class Skill:
    name: str           # max 64 chars, lowercase alphanumeric + hyphens
    description: str    # what the LLM sees when deciding to invoke it; max 1024 chars
    compatibility: str  # optional — environment requirements
    metadata: dict      # optional — arbitrary str→str key-value pairs
    allowed_tools: list # optional — tools this skill may invoke

    def run(self, input: dict, session: Session) -> dict: ...
```

---

## REST API

Applications are created and managed through the REST API. Configuration lives in a YAML file bundled as a ZIP with any referenced prompt templates and JSON schemas.

### App generator

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/generate` | Start a generation session from a natural-language description; returns `session_id` + `draft_config` |
| `GET` | `/generate/{session_id}` | Retrieve the current draft config for a session |
| `POST` | `/generate/{session_id}/revise` | Send a follow-up instruction to revise the draft |
| `POST` | `/generate/{session_id}/deploy` | Deploy the current draft as a new application |

### Application lifecycle

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/applications` | Create an application from a ZIP bundle |
| `GET` | `/applications` | List all applications |
| `GET` | `/applications/{name}` | Get application metadata |
| `PATCH` | `/applications/{name}` | Update config and restart |
| `DELETE` | `/applications/{name}` | Remove an application |

### Document ingestion and query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/applications/{name}/ingest_documents` | Ingest a batch of documents |
| `POST` | `/applications/{name}/query` | Answer a query (blocking) |
| `POST` | `/applications/{name}/query/stream` | Stream query response as Server-Sent Events |

### Workflows

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/workflows` | List registered workflow names |
| `POST` | `/applications/{name}/workflows/{workflow_name}/run` | Run a workflow (blocking); returns `{"workflow": "...", "records": [...], "total": N}` |
| `POST` | `/applications/{name}/workflows/{workflow_name}/stream` | Run a workflow, stream records as SSE |

### Skills management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/skills` | List skills assigned to an application |
| `POST` | `/applications/{name}/skills` | Assign a skill to an application |
| `DELETE` | `/applications/{name}/skills/{skill}` | Remove a skill from an application |
| `GET` | `/skills` | List all skills in the system registry |

### Adaptive evolution

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/suggestions` | List pending suggestions with supporting evidence (example queries, score distributions, session IDs) |
| `POST` | `/applications/{name}/suggestions/{id}/accept` | Accept a suggestion; triggers config patch + targeted re-ingest |
| `POST` | `/applications/{name}/suggestions/{id}/reject` | Reject a suggestion |

### Application config format

Applications are configured via a `config.yaml` inside a ZIP bundle. Any files referenced by filename
(JSON schemas, prompt templates) must also be present flat at the ZIP root.

For the annotated reference bundle, see
[`api/example_config.yaml`](https://github.com/CogBaseAI/cogbase/blob/main/api/example_config.yaml).
For a fully working zipped example, see the `_CONFIG_YAML` string in
[`examples/contract_analyst_demo/demo.py`](https://github.com/CogBaseAI/cogbase/blob/main/examples/contract_analyst_demo/demo.py), and 
[`examples/contract_compliance_demo/demo.py`](https://github.com/CogBaseAI/cogbase/blob/main/examples/contract_compliance_demo/demo.py).

---

## Quickstart

```bash
git clone https://github.com/cogbase/cogbase
cd cogbase
docker compose up
```

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

The contract analyst demo ingests legal contracts, extracts structured records (parties, dates, payment terms, key clauses), and supports full structured lookup and semantic search over the ingested portfolio.

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

About 60% of the codebase — the ingestion pipeline, contradiction engine, query runner, skill registry, memory layer, and store interfaces — is identical across all verticals. You write the config and schema once. The store adapters handle the rest.

---

## Project structure

```
cogbase/
├── cogbase/
│   ├── pipeline/             # Knowledge Pipeline
│   │   ├── ingestion/        # chunkers (fixed-size, langchain)
│   │   ├── extraction/       # LLM extractor base + implementation
│   │   └── ingestion_pipeline.py  # ChunkCollection, StructuredCollection, SummarizeCollection
│   ├── stores/               # Store adapter interfaces + built-in adapters
│   │   ├── base.py           # StructuredStoreBase, VectorStoreBase, VectorCollectionSchema
│   │   ├── schema.py         # CollectionSchema, FieldSchema, FieldType
│   │   ├── filters.py        # Filter, Op
│   │   ├── structured/       # SQLite, Postgres, in-memory
│   │   └── vector/           # FAISS, pgvector
│   ├── skills/               # Skill base class + registry
│   ├── embeddings/           # EmbeddingBase, OpenAI, HuggingFace
│   ├── llms/                 # LLMBase, OpenAI-compatible
│   ├── tools/                # Built-in tools (chunk-embed-upsert, extract)
│   ├── workflows/            # Workflow engine
│   │   ├── runner.py         # WorkflowRunner — sequential step executor
│   │   ├── context.py        # Jinja2 NativeEnvironment template rendering
│   │   └── tools/            # structured-query, vector-search, llm-structured, structured-save
│   └── core/                 # CogBaseApp, Runner, Session, models
├── api/                      # FastAPI REST API
│   ├── main.py               # App lifecycle, router registration
│   ├── config.py             # AppConfig (YAML schema)
│   ├── factory.py            # build_app from config
│   ├── routers/
│   │   ├── applications.py   # CRUD + ingest + query endpoints
│   │   └── skills.py         # System skill registry endpoint
│   ├── example_config.yaml   # Annotated config reference
│   └── example_system_config.yaml
├── examples/
│   └── contract_analyst_demo/
├── docker-compose.yml
└── README.md
```

---

## Roadmap

- [x] Core ingestion pipeline (chunk-embed-upsert, extract-structured, document-embed-upsert)
- [x] Typed fact extraction with configurable JSON schema
- [x] Store adapter interfaces (StructuredStoreBase, VectorStoreBase)
- [x] Built-in adapters: SQLite, Postgres, FAISS, pgvector
- [x] Per-document summarization vector collection
- [x] LLM agent query loop with structured_lookup and vector_search tools
- [x] Skill registry + base skill interface
- [x] REST API (create/update/delete apps, ingest, query, streaming)
- [x] Contract analyst example
- [x] Declarative workflow engine (structured-query, vector-search, llm-structured, structured-save; foreach loops; after_ingest triggers)
- [ ] App generator (conversational config generation from description + example questions, with iterative revision)
- [ ] Contradiction detection engine (date, numeric, statement conflicts)
- [ ] Short-term memory (Redis + in-memory)
- [ ] Episodic memory (conversation + agent history)
- [ ] Long-term memory (cross-session knowledge store)
- [ ] Adaptive evolution engine (gap detector: retrieval score analysis, null-answer pattern mining, tool-chain clustering)
- [ ] Suggestion surface API (GET /suggestions, accept/reject with targeted re-ingest)
- [ ] Docker Compose quickstart
- [ ] Insurance example
- [ ] Medical records example
- [ ] Managed cloud hosting (SOC 2)

---

## Contributing

CogBase is in early development. The best way to contribute right now:

- **Try the quickstart** and file issues for anything that breaks
- **Contribute a store adapter** — implement `StructuredStoreBase` or `VectorStoreBase` for a backend not yet supported
- **Contribute an example** — a YAML config + JSON schema + prompt file for a new vertical
- **Contribute a skill** — new capabilities that implement the skill interface are always welcome
- **Improve the contradiction engine** — it's the hardest and most valuable part; PRs with test cases are especially welcome
- **Improve the memory layer** — especially long-term memory retrieval and conflict resolution across sessions

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0
