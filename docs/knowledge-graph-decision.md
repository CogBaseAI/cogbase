# Knowledge Graph Decision

## Decision

CogBase does not include an explicit knowledge graph (KG) layer by default.

For the target workload — enterprise document AI over legal, finance, insurance, research, and similar corpora — CogBase uses structured extraction, vector retrieval, document reading, and an LLM agent loop instead of precomputing an entity-and-edge graph.

This decision can be revisited if product requirements shift toward exhaustive graph traversal, global graph analytics, or strict completeness guarantees over implicit relationships.

## Context

Knowledge graphs are useful when an application needs explicit entities, typed relationships, canonical identifiers, and deterministic traversal. They also introduce substantial implementation and maintenance costs:

- ontology design
- entity extraction and canonicalization
- relationship extraction and typing
- edge refresh as source documents change
- graph storage, query planning, and operational maintenance

CogBase optimizes for lower setup cost and practical document-grounded reasoning rather than exhaustive relationship modeling.

## Architecture Used Instead

Documents flow into two primary stores:

- **Vector store** — passage chunks and document summaries for semantic retrieval
- **Structured store** — LLM-extracted typed records for exact filters, comparison, and aggregation

At query time, an LLM agent loop calls retrieval and inspection tools:

- `structured_lookup` — exact record queries with field filters
- `vector_search` — semantic search over any configured collection
- `read_document` — broader context from a document's original text
- skill tools — application-registered capabilities that extend the agent beyond built-in retrieval

When the agent retrieves a chunk, it can inspect the entities and facts in that chunk, issue follow-up searches, read related documents, and iterate until it has enough evidence. This approximates short graph traversal without requiring a precomputed graph.

The structured store is intentionally separate from the vector store. Many questions that appear to require a graph are actually typed-record queries: filter contracts by renewal date, compare exposure by counterparty, aggregate policy limits, or list claims above a threshold.

## Alternative Considered: Explicit KG Layer

An explicit KG would model entities and relationships during ingestion, then answer relationship queries through graph traversal.

It provides stronger support for:

- deterministic multi-hop traversal
- canonical entity disambiguation
- stable relationship typing
- global graph algorithms
- completeness guarantees over known edges

CogBase does not adopt this as the default because these benefits are not required for most target document workflows, while the upfront modeling and maintenance costs are high.

## Coverage And Tradeoffs

**Multi-hop traversal.** A KG can traverse A->B->C in a single graph query. CogBase performs this through iterative search and reading. This works well for typical 2-4 hop workflows, but becomes slower and less reliable for long chains.

**Entity disambiguation.** A KG can resolve ambiguous entities through canonical nodes. CogBase resolves ambiguity at inference time from document context. This is usually sufficient for enterprise corpora with consistent terminology, but weaker for noisy multi-source corpora.

**Relationship typing.** A KG stores explicit relationship types such as `causes`, `contradicts`, or `subsumes`. CogBase lets the LLM infer relationship type from retrieved evidence at answer time. This is more flexible but less consistent across sessions.

**Global graph algorithms.** PageRank, community detection, shortest path, and similar full-corpus algorithms require a graph. CogBase does not support these as native operations.

**Completeness guarantees.** KG traversal can exhaustively return all known edges matching a predicate. CogBase retrieval is probabilistic and agent-directed. CogBase does not currently provide completeness guarantees over implicit relationships.

**Ontology and inheritance.** Domain hierarchies can be supplied through prompts, schemas, structured extraction rules, or skill tools. This covers many practical vertical workflows, but it is not a substitute for a formally maintained ontology when one is required.

## Memory And Adaptive Evolution

CogBase can recover some KG-like benefits through usage-driven learning.

The memory layer can record successful retrieval paths. For example, an indemnification exposure query might repeatedly follow this path:

`vector_search` -> `read_document` -> `structured_lookup` for counterparty litigation history

When similar queries recur, memory can replay the successful path and avoid rediscovering it from scratch.

The adaptive evolution engine can materialize frequently needed cross-references into the structured store. If users repeatedly query contract terms and then look up litigation history for the same counterparty, CogBase can propose a foreign-key-style relationship between those collections. That relationship functions like a graph edge, but it is derived from observed usage rather than upfront ontology design.

## When To Add A KG

Add or integrate an explicit KG when the application requires:

- **Completeness-critical relationship queries** — drug interaction safety, AML fraud detection, regulatory compliance, or other workflows where a missed edge creates legal, financial, or safety risk
- **Long chain traversal** — 6+ hops at scale across large node sets
- **Global graph algorithms** — centrality, community detection, shortest path, influence analysis, or corpus-wide relationship analytics
- **Strict canonical entity management** — noisy, multi-source corpora where inference-time disambiguation is not reliable enough
- **Formal ontology governance** — domain rules where inheritance, relationship typing, and schema evolution must be explicitly controlled

## Revisit Criteria

Revisit this decision if any of the following become common product requirements:

- users need exhaustive relationship queries rather than evidence-backed answers
- retrieval paths regularly exceed 4 hops
- repeated cross-references cannot be handled through structured-store evolution
- global graph algorithms become part of the product surface
- customers require auditable completeness guarantees over implicit relationships
- entity ambiguity causes recurring answer quality failures

Until those conditions appear, structured extraction plus LLM-driven retrieval is the preferred default for CogBase because it covers the common enterprise document AI workflows with lower engineering and maintenance cost.
