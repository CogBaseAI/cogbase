# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

CogBase is in early development. `cogbase/core/` and `cogbase/stores/base.py` are implemented; all other architecture described below is planned.

## Architecture

CogBase is a framework for structured fact extraction, contradiction detection, and grounded LLM reasoning over large document sets. It has four layers with clean boundaries:

**Knowledge Pipeline** (async, ingest-time)
- Parses raw inputs (PDF, DOCX, email, chat, transcripts)
- Runs typed fact extraction via domain pack configuration
- Writes structured facts to the structured store and vector embeddings to the vector store

**Reasoning Engine** (real-time, query-time)
- Query router classifies intent into four patterns before touching any store:
  - Pattern A: structured lookup (never hits LLM)
  - Pattern B: semantic vector search
  - Pattern C: hybrid retrieval from both stores + reasoning
  - Pattern D: grounded generation (separates `[FACTS]` from `[SUPPORTING_QUOTES]`)
- All capabilities are exposed as composable, stateless Skills

**Agent Layer** (goal-driven)
- Agents plan skill sequences, execute, observe, and replan
- Reads/writes episodic memory to resume across sessions

**Memory Layer** (persistent)
- Short-term: Redis-backed session context
- Episodic: conversation + agent action history in structured store
- Long-term: cross-session confirmed facts, resolved contradictions, preferences

## Planned project structure

```
cogbase/
├── cogbase/
│   ├── pipeline/         # ingestion/, extraction/, contradiction/
│   ├── engine/           # router/, retrieval/, generation/
│   ├── memory/           # short_term.py, episodic.py, long_term.py
│   ├── stores/           # base.py (StructuredStoreBase, VectorStoreBase), structured/, vector/
│   ├── skills/           # built-in skill definitions
│   ├── agents/           # built-in agents
│   └── core/             # skill registry, session, base classes
├── packs/                # domain config packs (YAML + prompts + optional skills/agents)
│   └── legal/
├── api/                  # REST API
└── docker-compose.yml
```

## Key interfaces

**Store adapters** — implement these to add a new backend:
```python
class StructuredStoreBase:
    def save_facts(self, facts: list[Fact]) -> None: ...
    def query_facts(self, filters: dict) -> list[Fact]: ...
    def save_timeline(self, events: list[Event]) -> None: ...
    def query_timeline(self, session_id: str) -> list[Event]: ...
    def save_contradiction(self, c: Contradiction) -> None: ...
    def query_contradictions(self, filters: dict) -> list[Contradiction]: ...

class VectorStoreBase:
    def upsert(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, query_embedding: list[float], top_k: int) -> list[Chunk]: ...
    def delete(self, doc_id: str) -> None: ...
```

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
Expected inputs/outputs are documented in each skill's class docstring or a `SKILL.md` alongside the implementation. `name` and `description` are validated at class-definition time.

**Fact schema** — every extracted fact carries: `type`, `value`, `raw_text`, `doc_id`, `page`, `confidence`. `raw_text` is preserved verbatim as the citation.

## Domain packs

Domain configuration lives in YAML + prompt templates under `packs/<domain>/`, not in Python code. Packs define what facts to extract, what contradictions to detect, and can ship their own skills and agents. The legal pack ships with the project; community packs are YAML + prompts + optional skills/agents.

## Contradiction detection approach

Two-phase (not a single LLM prompt over long context):
1. Extract typed facts from each source individually
2. Cross-document comparison using embedding distance + NLI classification, bucketed by conflict type (date, numeric, statement)

Previously resolved contradictions are stored in long-term memory and excluded from future scans.

## Quickstart (planned)

```bash
git clone https://github.com/cogbase/cogbase
cd cogbase
docker compose up
```
