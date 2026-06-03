# Concepts

## App generator

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

## Structured extraction

Every document is processed into structured records at ingestion time. Extraction is general — any JSON schema works: facts, entities, clauses, events, relationships, risk flags, and more. Each extractor declares the collection it writes to and its schema.

---

## Per-document summarization

Alongside passage chunks, the pipeline supports a `document-embed-upsert` step that generates an LLM summary per document and stores its embedding as a single vector. This gives the query runner two levels of semantic retrieval:

- **document_chunks** — precise, passage-level retrieval for detailed or specific questions
- **document_summary** — topic-level retrieval for high-level questions about what documents cover

The LLM automatically picks the right collection based on the query.

---

## Workflows

Workflows are named, YAML-declared analytical pipelines that run over already-ingested collections. They compose four built-in tools in any sequence, including `foreach` loops over result sets:

| Tool | What it does |
|---|---|
| `structured-query` | Read typed records with equality filters; result at `steps.<id>.records` |
| `vector-search` | Embed a query string and search a vector collection; result at `steps.<id>.chunks` |
| `llm-structured` | Call the LLM with a system prompt and JSON input, validate against a JSON Schema; result at `steps.<id>.output` |
| `structured-save` | Upsert records into a collection and stream each one to the caller; result at `steps.<id>.records` |

Step parameters are Jinja2 templates with three namespaces: `input` (invocation params), `steps.<id>` (prior step outputs), and `item` (current foreach element). A `{{ expr }}` that resolves to a list returns an actual Python list, not a string.

Workflows can be triggered manually via `POST /applications/{name}/workflows/{workflow_name}/run` or automatically after each successful document ingest (`trigger.type: after_ingest`, optionally gated by document metadata). Blocking and streaming (`/stream`) endpoints are both available.

---

## LLM agent query loop

The query runner drives a multi-turn LLM agent loop with configurable retrieval tools:

| Tool | Description |
|---|---|
| `structured_lookup` | Exact record query against a named collection with field filters |
| `vector_search` | Semantic search against a named vector collection (chunks or summaries) |
| `read_document` | Fetch a slice of a document's original text by character offset; used to get broader context around a retrieved chunk |
| `skill tools` | Custom capabilities registered with the application |

The LLM calls tools as needed to gather evidence, then synthesises a grounded answer. No fixed routing pattern — the model decides. When `structured_lookup` returns a large result set (above the passthrough token threshold), records are returned directly as formatted text without an additional synthesis step.

A default `system_prompt` can be set per application in `config.yaml` and overridden per request via `QueryRequest.system_prompt`. Query responses include `input_tokens` and `output_tokens` totals across all LLM calls.

---

## Pluggable stores

CogBase defines clean adapter interfaces for both stores. Swap backends via config — no application code changes required.

```python
from cogbase.stores import StructuredStoreBase, VectorStoreBase, CollectionSchema, VectorCollectionSchema, Filter
from cogbase.core.models import Chunk

class MyStructuredStore(StructuredStoreBase):
    async def create_collection(self, schema: CollectionSchema) -> None: ...
    async def save(self, collection: str, records: list[dict]) -> None: ...
    async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]: ...
    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

class MyVectorStore(VectorStoreBase):
    async def upsert(self, collection: str, chunks: list[Chunk]) -> None: ...
    async def search(self, collection: str, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]: ...
    async def delete(self, collection: str, doc_id: str) -> None: ...
```

Built-in adapters: SQLite + FAISS (local/dev), Postgres + pgvector (production).

---

## Memory

CogBase maintains three tiers of memory, each scoped and persisted differently:

| Tier | Scope | Purpose |
|---|---|---|
| Short-term | Session | Assembled context window for the current query; expires with the session |
| Episodic | User / session | Full history of queries, answers, and agent actions; enables follow-ups and agent continuity |
| Long-term | User / project / org | Confirmed facts, learned patterns, preferences; persists indefinitely |

---

## Adaptive evolution

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

---

## Skills

Skills are the unit of custom capability in CogBase — discrete, stateless, and composable. Each skill is a Python class with a `name`, `description` (shown to the LLM when selecting tools), and a `run(input, session)` method. Skills are registered with an application via the REST API and appear as callable tools in the query runner's agent loop.
