# CogBase

**Ingest anything. Extract structured facts. Reason across all of it.**

CogBase is an open-source framework for building AI applications that need to understand, cross-reference, and reason over large volumes of unstructured data — documents, emails, transcripts, chat logs, reports, and more.

It provides the foundational layer that vertical AI products are built on: typed fact extraction, contradiction detection, a pluggable hybrid store, intelligent query routing, composable skills, goal-driven agents, and a multi-tier memory system — all configurable for any domain.

---

## The problem

Most RAG pipelines retrieve text and pass it to an LLM. That works for simple Q&A. It breaks down when you need to:

- Spot contradictions between two sources ("the contract says 60 days; the email says 30")
- Build a reliable timeline across dozens of documents
- Answer questions that require reasoning over structured facts, not just semantic similarity
- Ground generated output in citable, auditable sources
- Automate multi-step workflows across a large document set
- Maintain continuity across sessions and accumulate knowledge over time

CogBase solves this with a structured extraction layer sitting between raw ingestion and the LLM — turning unstructured input into typed, queryable facts before any reasoning begins — an agent layer that composes skills into goal-driven workflows, and a memory layer that persists knowledge across queries, sessions, and time.

---

## Architecture

CogBase is organized into four layers with clean boundaries between them.

```
╔═══════════════════════════════════════════════════════════╗
║  KNOWLEDGE PIPELINE                        (async)        ║
║                                                           ║
║  Raw inputs                                               ║
║  (PDF, DOCX, email, chat, transcript, ...)                ║
║          ↓                                                ║
║  Ingestion & parsing                                      ║
║          ↓                                                ║
║  Structured fact extraction  ←  domain pack               ║
║          ↓                ↓                               ║
║  ┌──────────────────┐  ┌──────────────────┐               ║
║  │ Structured Store │  │   Vector Store   │               ║
║  │ (facts, timeline │  │ (semantic chunks)│               ║
║  │  entities, risks │  │                  │               ║
║  │  contradictions) │  │                  │               ║
║  └──────────────────┘  └──────────────────┘               ║
╚═══════════════════════════════════════════════════════════╝
          ↕  hybrid store              ↕  writes long-term memory
╔═══════════════════════════════════════════════════════════╗
║  REASONING ENGINE                          (real-time)    ║
║                                                           ║
║  User query                                               ║
║          ↓                                                ║
║  Query router                                             ║
║  (A: structured · B: semantic · C: hybrid · D: generate)  ║
║          ↓                                                ║
║  Skills  (atomic, composable capabilities)                ║
║          ↓                                                ║
║  LLM reasoning layer  ←  short-term memory (context)      ║
║          ↓                                                ║
║  Grounded, cited response                                 ║
╚═══════════════════════════════════════════════════════════╝
          ↕  skill registry             ↕  reads/writes episodic memory
╔═══════════════════════════════════════════════════════════╗
║  AGENT LAYER                               (goal-driven)  ║
║                                                           ║
║  Goal                                                     ║
║          ↓                                                ║
║  Agent plans skill sequence  ←  episodic memory           ║
║          ↓                                                ║
║  Execute → observe → replan if needed                     ║
║          ↓                                                ║
║  Result                                                   ║
╚═══════════════════════════════════════════════════════════╝
          ↕  read/write across all layers
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

**Knowledge Pipeline** runs asynchronously at ingest time. Raw inputs stay in blob storage. Extracted structured data — facts, timeline events, entities, risk flags, contradictions — goes into the structured store. Vector embeddings of chunked text go into the vector store for semantic search. Both stores are pluggable — swap backends without changing application code.

**Reasoning Engine** runs in real-time at query time. A query router classifies intent before touching either store — Pattern A questions are pure structured lookups and never reach the LLM. Pattern C and D queries assemble grounded context from both stores and short-term memory before passing it to the reasoning layer. Every capability in the engine is exposed as a composable skill.

**Agent Layer** sits on top of the Reasoning Engine. Agents receive a goal, access the skill registry, and use the LLM to plan and execute a sequence of skills — replanning if intermediate results change the approach. Agents read and write episodic memory so multi-step tasks can resume across sessions without starting from zero.

**Memory Layer** serves all three layers above. Short-term memory holds the assembled context for the current query. Episodic memory logs the full history of queries, answers, and agent actions. Long-term memory accumulates confirmed facts, resolved contradictions, learned patterns, and user preferences across sessions.

---

## Core capabilities

### Structured extraction

Every document is processed into structured records at ingestion time. Extraction is general — any Pydantic model works: facts, entities, clauses, events, relationships, risk flags, and more. Each extractor declares the collection it writes to and its schema; domain packs define their own record types without touching core code.

The built-in `Fact` model carries: `type`, `value`, `raw_text`, `doc_id`, `page`, `confidence`. The `raw_text` field is preserved verbatim from the source and used as the citation.

### Contradiction detection

CogBase uses a two-phase approach rather than a single "find contradictions" prompt, which is unreliable over long context:

1. Extract typed facts from each source individually
2. Run a cross-document comparison pass over the fact store, using embedding distance + NLI classification to flag conflicts by type (date conflicts, numeric conflicts, statement conflicts)

This makes contradiction detection a query over structured data, not a needle-in-a-haystack prompt. Previously resolved contradictions are stored in long-term memory and not re-flagged in future sessions.

### Typed query routing

Before touching either store, an intent classifier decides which execution path applies:

| Pattern | Description | Example |
|---|---|---|
| A — Structured lookup | Answer from structured store directly | "How many days notice was given?" |
| B — Semantic retrieval | Vector search over embedded chunks | "What did Chen say about her own performance?" |
| C — Hybrid reasoning | Retrieve from both stores, reason over results | "Does the review contradict the termination reason?" |
| D — Grounded generation | Retrieve structured results + quotes, then draft | "Draft a demand letter using these facts" |

Pattern A questions never touch the LLM. Pattern C is where orchestration work lives. Pattern D separates the `[FINDINGS]` block (structured results from the structured store) from the `[SUPPORTING_QUOTES]` block (verbatim text from the vector store) so every generated claim is auditable.

### Pluggable stores

CogBase defines clean adapter interfaces for both stores. Swap backends via config — no application code changes required.

```python
# Default — Postgres + pgvector
cog = CogBase(pack="legal")

# Mix and match backends
cog = CogBase(
    pack="legal",
    structured_store="mongodb",
    vector_store="pinecone",
)

# Bring your own adapter
from cogbase.stores import StructuredStoreBase, VectorStoreBase

class MyStructuredStore(StructuredStoreBase):
    async def create_collection(self, schema: CollectionSchema) -> None: ...
    async def save(self, collection: str, records: list[BaseModel]) -> None: ...
    async def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]: ...
    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

class MyVectorStore(VectorStoreBase):
    async def upsert(self, chunks: list[Chunk]) -> None: ...
    async def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]: ...
    async def delete(self, doc_id: str) -> None: ...

cog = CogBase(
    structured_store=MyStructuredStore(),
    vector_store=MyVectorStore(),
)
```

### Memory

CogBase maintains three tiers of memory, each scoped and persisted differently:

| Tier | Scope | Purpose |
|---|---|---|
| Short-term | Session | Assembled context window for the current query; expires with the session |
| Episodic | User / session | Full history of queries, answers, and agent actions; enables follow-ups and agent continuity |
| Long-term | User / project / org | Confirmed facts, resolved contradictions, learned patterns, preferences; persists indefinitely |

```python
# Short-term: context for the current query is assembled automatically
result = cog.query("What was the notice period?")

# Episodic: follow-up questions work across turns
result = cog.query("And did they comply with it?")  # knows what "it" refers to

# Long-term: confirmed facts persist across sessions
cog.memory.confirm("notice_period_was_45_days", source=result.citations)

# Next session — no need to re-query
cog.memory.recall("notice_period")  # returns confirmed fact instantly
```

### Skills

Skills are the atomic unit of capability in CogBase — discrete, stateless, and composable. Every built-in capability is a registered skill. Contributors and domain packs can add new ones without touching core code.

**Built-in skills:**

| Skill | What it does |
|---|---|
| `ingest` | Parse and chunk a document or data source |
| `extract_facts` | Run typed fact extraction on a chunk |
| `detect_contradictions` | Compare facts across sources |
| `query_structured` | Query the structured store |
| `query_semantic` | Vector search over embeddings |
| `query_hybrid` | Combined structured + semantic retrieval |
| `build_timeline` | Assemble chronological event sequence |
| `flag_risks` | Identify risk patterns from facts |
| `summarize` | Grounded summary of retrieved context |
| `draft` | Grounded generation from structured results + supporting quotes |
| `remember` | Write a confirmed fact or conclusion to long-term memory |
| `recall` | Retrieve from long-term memory by key or semantic search |

Every skill shares a consistent interface, aligned with the [AgentSkills specification](https://agentskills.io/specification):

```python
class Skill:
    name: str           # max 64 chars, lowercase alphanumeric + hyphens
    description: str    # what the LLM sees when deciding to invoke it; max 1024 chars
    compatibility: str  # optional — environment requirements
    metadata: dict      # optional — arbitrary str→str key-value pairs
    allowed_tools: list # optional — tools this skill may invoke

    def run(self, input: dict, session: Session) -> dict: ...
```

Expected inputs and outputs are documented in each skill's class docstring or a `SKILL.md` file alongside the implementation.

### Agents

Agents orchestrate skills to accomplish multi-step goals. They plan a skill sequence, execute it, observe results, and replan if needed. Episodic memory means agents can resume across sessions without losing context.

**Built-in agents:**

| Agent | What it does |
|---|---|
| `ResearchAgent` | Plans and chains retrieval skills to answer a complex question |
| `ContradictionAgent` | Proactively scans a session for all conflicts across sources |
| `DraftingAgent` | Retrieves structured results + supporting quotes, then generates a grounded document |
| `IngestionAgent` | Watches a folder or source, ingests new files, updates the store |
| `DiligenceAgent` | Systematically works through a document set flagging risks and gaps |

### Domain packs

Domain configuration lives in YAML files and prompt templates — not Python code. Switching verticals means changing what the extraction prompt looks for, not rewriting the pipeline. Packs can also ship their own skills and agents.

```
packs/
├── legal/
│   ├── facts.yaml
│   ├── contradictions.yaml
│   ├── prompts/
│   ├── skills/
│   └── agents/
├── insurance/
├── medical/
└── compliance/
```

The legal pack ships with the project. Community packs are contributed as YAML + prompt files + optional skills and agents.

---

## Quickstart

```bash
git clone https://github.com/cogbase/cogbase
cd cogbase
docker compose up
```

Upload documents, select a pack, and start querying — all in under 10 minutes.

```python
from cogbase import CogBase

cog = CogBase(pack="legal")

# Ingest documents
cog.ingest("./case_documents/")

# Query — router picks the execution path automatically
result = cog.query("Does the performance review contradict the stated reason for termination?")

print(result.answer)
print(result.citations)       # source documents + page references
print(result.contradictions)  # any flagged conflicts

# Follow-up using episodic memory
result = cog.query("Which document is the stronger evidence?")

# Confirm a fact to long-term memory
cog.memory.confirm("termination_was_pretextual", source=result.citations)

# Run an agent for a multi-step goal
agent = cog.agent("DiligenceAgent")
report = agent.run("Identify every material risk in this document set")
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

About 60% of the codebase — the ingestion pipeline, contradiction engine, query router, skill registry, memory layer, and store interfaces — is identical across all verticals. You write it once. The domain pack and store adapters handle the rest.

---

## Project structure

```
cogbase/
├── cogbase/
│   ├── pipeline/             # Knowledge Pipeline
│   │   ├── ingestion/        # parsers for PDF, DOCX, email, chat, etc.
│   │   ├── extraction/       # typed fact extraction
│   │   └── contradiction/    # contradiction detection engine
│   ├── engine/               # Reasoning Engine
│   │   ├── router/           # query intent classifier
│   │   ├── retrieval/        # structured + semantic + hybrid execution
│   │   └── generation/       # grounded generation layer
│   ├── memory/               # Memory Layer
│   │   ├── short_term.py     # Redis-backed session context
│   │   ├── episodic.py       # conversation + agent action history
│   │   └── long_term.py      # cross-session facts, conclusions, preferences
│   ├── stores/               # Store adapter interfaces + built-in adapters
│   │   ├── base.py           # StructuredStoreBase, VectorStoreBase
│   │   ├── structured/
│   │   └── vector/
│   ├── skills/               # built-in skill definitions + implementations
│   ├── agents/               # built-in agents
│   └── core/                 # skill registry, session, base classes
├── packs/                    # domain configuration packs
│   └── legal/
│       ├── facts.yaml
│       ├── contradictions.yaml
│       ├── prompts/
│       ├── skills/
│       └── agents/
├── api/                      # REST API
├── docker-compose.yml
└── README.md
```

---

## Roadmap

- [ ] Core ingestion pipeline (PDF, DOCX, email, chat export)
- [ ] Typed fact extraction with configurable schema
- [ ] Store adapter interfaces (StructuredStoreBase, VectorStoreBase)
- [ ] Contradiction detection engine (date, numeric, statement conflicts)
- [ ] Query router (rule-based pre-filter + LLM classifier)
- [ ] Short-term memory (Redis + in-memory)
- [ ] Episodic memory (conversation + agent history)
- [ ] Long-term memory (cross-session knowledge store)
- [ ] Skill registry + base skill interface
- [ ] Built-in skills (ingest, extract_facts, query_*, draft, remember, recall, ...)
- [ ] Built-in agents (ResearchAgent, ContradictionAgent, DraftingAgent, ...)
- [ ] Legal domain pack
- [ ] REST API + Python SDK
- [ ] Docker Compose quickstart
- [ ] Insurance pack
- [ ] Medical records pack
- [ ] Managed cloud hosting (SOC 2)

---

## Contributing

CogBase is in early development. The best way to contribute right now:

- **Try the quickstart** and file issues for anything that breaks
- **Contribute a store adapter** — implement `StructuredStoreBase` or `VectorStoreBase` for a backend not yet supported
- **Contribute a domain pack** — if you work in insurance, medical, compliance, or M&A, a YAML config + prompt file is all it takes
- **Contribute a skill or agent** — new capabilities that implement the base interface are always welcome
- **Improve the contradiction engine** — it's the hardest and most valuable part; PRs with test cases are especially welcome
- **Improve the memory layer** — especially long-term memory retrieval and conflict resolution across sessions

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0
