# CogBase

**Ingest anything. Extract structured facts. Reason across all of it.**

CogBase is an open-source framework for building AI applications that need to understand, cross-reference, and reason over large volumes of unstructured data — documents, emails, transcripts, chat logs, reports, and more.

It provides the foundational layer that vertical AI products are built on: typed fact extraction, contradiction detection, a pluggable hybrid store, an LLM agent query runner, composable skills, and a multi-tier memory system — all configurable for any domain through a REST API.

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

CogBase is organized into three layers with clean boundaries between them.

```
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
║    summarize-embed-upsert → one summary vector/doc        ║
║          ↓                ↓                ↓              ║
║  ┌──────────────────┐  ┌──────────────────────────────┐   ║
║  │ Structured Store │  │       Vector Store           │   ║
║  │ (typed records,  │  │ document_chunks (passages)   │   ║
║  │  schemas, facts) │  │ document_summary (per-doc)   │   ║
║  └──────────────────┘  └──────────────────────────────┘   ║
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
```

**Knowledge Pipeline** runs asynchronously at ingest time. Three step types can be combined in any order:

- `chunk-embed-upsert` — splits document text into overlapping passages, embeds them, and upserts into a vector collection for passage-level semantic search
- `extract-structured` — runs a configurable LLM extractor to produce typed records stored in a structured collection
- `summarize-embed-upsert` — generates one LLM summary per document, embeds it, and upserts into a vector collection for document-level semantic search

Both stores are pluggable — swap backends without changing application code.

**Query Runner** drives a real-time LLM agent loop. Rather than a fixed routing pattern, the LLM receives the available tools and decides which to call: `structured_lookup` for exact record queries, `vector_search` against any configured vector collection (passage chunks, document summaries, or both), and any skill tools registered with the application. The loop continues until the LLM has enough evidence to produce a final answer. Large structured result sets are returned directly without synthesis (passthrough rule).

**Memory Layer** serves the layers above. Short-term memory holds the assembled context for the current query. Episodic memory logs the full history of queries, answers, and agent actions. Long-term memory accumulates confirmed facts, resolved contradictions, learned patterns, and user preferences across sessions.

---

## Core capabilities

### Structured extraction

Every document is processed into structured records at ingestion time. Extraction is general — any JSON schema works: facts, entities, clauses, events, relationships, risk flags, and more. Each extractor declares the collection it writes to and its schema.

The built-in `Fact` model carries: `type`, `value`, `raw_text`, `doc_id`, `page`, `confidence`. The `raw_text` field is preserved verbatim from the source and used as the citation.

### Per-document summarization

Alongside passage chunks, the pipeline supports a `summarize-embed-upsert` step that generates one LLM summary per document and stores its embedding as a single vector. This gives the query runner two levels of semantic retrieval:

- **document_chunks** — precise, passage-level retrieval for detailed or specific questions
- **document_summary** — topic-level retrieval for high-level questions about what documents cover

The LLM automatically picks the right collection based on the query.

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
| skill tools | Custom capabilities registered with the application |

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

### Skills management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/skills` | List skills assigned to an application |
| `POST` | `/applications/{name}/skills` | Assign a skill to an application |
| `DELETE` | `/applications/{name}/skills/{skill}` | Remove a skill from an application |
| `GET` | `/skills` | List all skills in the system registry |

### Application config format

Applications are configured via a `config.yaml` inside a ZIP bundle. Any files referenced by filename (JSON schemas, prompt templates) must also be present flat at the ZIP root.

```yaml
name: my-contract-analyzer

llm:
  provider: openai
  model: gpt-4o-mini
  # api_key: sk-...          # omit to use OPENAI_API_KEY env var

embedding:                   # shared across all vector and summarize collections
  provider: openai
  model: text-embedding-3-small

vector_collections:
  - name: document_chunks
    chunker:
      type: fixed
      chunk_size: 512
      overlap: 64

structured_collections:
  - name: contract_extraction
    schema: extraction_schema.json      # filename in ZIP root
    extractor:
      type: llm
      prompt: extraction_prompt.txt     # filename in ZIP root; omit for built-in default

summarize_collections:
  - name: document_summary              # one summary vector per document
    prompt: "Summarize this document in a few sentences."
    max_tokens: 1024

pipeline:
  steps:
    - tool: chunk-embed-upsert
      collection: document_chunks
    - tool: extract-structured
      collection: contract_extraction
    - tool: summarize-embed-upsert
      collection: document_summary
```

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

- [x] Core ingestion pipeline (chunk-embed-upsert, extract-structured, summarize-embed-upsert)
- [x] Typed fact extraction with configurable JSON schema
- [x] Store adapter interfaces (StructuredStoreBase, VectorStoreBase)
- [x] Built-in adapters: SQLite, Postgres, FAISS, pgvector
- [x] Per-document summarization vector collection
- [x] LLM agent query loop with structured_lookup and vector_search tools
- [x] Skill registry + base skill interface
- [x] REST API (create/update/delete apps, ingest, query, streaming)
- [x] Contract analyst example
- [ ] Contradiction detection engine (date, numeric, statement conflicts)
- [ ] Short-term memory (Redis + in-memory)
- [ ] Episodic memory (conversation + agent history)
- [ ] Long-term memory (cross-session knowledge store)
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
