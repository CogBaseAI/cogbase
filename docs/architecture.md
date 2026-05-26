# Architecture

CogBase is organized into six layers with clean boundaries between them.

## App Generator

The App Generator is the entry point for new applications. Instead of writing `config.yaml` by hand, describe your documents and example questions in natural language and the system generates the full configuration — collections, steps, schemas, prompts, and workflows — as a draft you can then revise conversationally before deploying.

## Knowledge Pipeline

The Knowledge Pipeline runs asynchronously at ingest time. Three step types can be combined in any order, with optional `when_meta` predicates to route specific document types to different steps:

- `chunk-embed-upsert` — splits document text into overlapping passages, embeds them, and upserts into a vector collection for passage-level semantic search
- `extract-structured` — runs a configurable LLM extractor to produce typed records stored in a structured collection
- `document-embed-upsert` — generates one vector such as LLM summary per document, embeds it, and upserts into a vector collection for document-level semantic search

Both stores are pluggable — swap backends without changing application code.

## Document Registry

The Document Registry tracks every document ingested into an application — its path, status, and timestamps. When a document needs to pass through a workflow after ingestion, a `DocWorkflowRecord` is created at ingest time and updated as the workflow progresses (`PENDING` → `READY` → `RUNNING` → `DONE` / `FAILED`). Ingest and workflow runs are tracked as idempotent `TaskRecord` entries: a task can be re-run at any time without side effects. The document store is uploaded before ingest begins so the task owns the full pipeline and can be retried independently of the upload.

## Workflows

Workflows run on-demand (via API call) or automatically after a successful ingest (`after_ingest` trigger). They are YAML-declared sequential pipelines over already-ingested collections — reading from structured and vector stores, calling an LLM to judge or classify, and writing derived records back to output collections. They stream results as SSE. Workflows sit between the pipeline (document-time) and skills (query-time, LLM-callable), handling analytical computations that need to fan out over many records but don't belong in the ingest step itself.

## Query Runner

The Query Runner drives a real-time LLM agent loop. Rather than a fixed routing pattern, the LLM receives the available tools and decides which to call: `structured_lookup` for exact record queries, `vector_search` against any configured vector collection (passage chunks, document summaries, or both), and any skill tools registered with the application. The loop continues until the LLM has enough evidence to produce a final answer. Large structured result sets are returned directly without synthesis (passthrough rule).

## Memory Layer

The Memory Layer serves the layers above. Short-term memory holds the assembled context for the current query. Episodic memory logs the full history of queries, answers, and agent actions. Long-term memory accumulates confirmed facts, learned patterns, and user preferences across sessions.

## Adaptive Evolution

Adaptive Evolution closes the feedback loop between usage and configuration. A background gap detector mines episodic logs for signals that the current config doesn't cover what users actually ask: low-scoring retrieval results suggest a missing collection or pipeline step; repeated "I don't have that information" answers suggest a missing structured field; recurring multi-step tool chains suggest a skill worth encapsulating. The system surfaces these as concrete, evidence-backed suggestions and waits for user confirmation before applying any change. On acceptance, the config is patched and only the affected documents are re-ingested.
