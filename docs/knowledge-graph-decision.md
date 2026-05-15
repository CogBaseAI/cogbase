# Why CogBase Does Not Use a Knowledge Graph

CogBase deliberately omits an explicit knowledge graph (KG) layer. This document explains that decision, where LLM-driven retrieval covers the same ground, and the narrow cases where a KG remains the right tool.

## The Architecture CogBase Uses Instead

Documents flow into a vector store (passage chunks and document summaries) and a structured store (LLM-extracted typed records). At query time an LLM agent loop calls three tools:

- `structured_lookup` — exact record queries with field filters
- `vector_search` — semantic search over any configured collection
- `read_document` — Read a broader context of a document's original text
- skill tools — custom capabilities registered with the application

When the LLM retrieves chunks, it sees the entities they contain, issues follow-up searches, and iterates until it has enough evidence to answer. This pattern approximates graph traversal without a pre-computed graph.

## What KGs Actually Provide — and How the Architecture Covers It

**Multi-hop traversal.** A KG walks A→B→C in a single query. In CogBase, the LLM issues a vector search, reads B, and issues another search for C. For 2–4 hops this works well. At 6+ hops, iterative retrieval becomes slower and probabilistically lossy.

**Entity disambiguation.** KGs resolve "Apple (company)" vs "Apple (fruit)" through canonical entity nodes. The LLM handles this at inference time from context. For most enterprise document sets with consistent terminology, inference-time resolution is good enough; for extremely noisy multi-source corpora it is less reliable.

**Relationship typing.** KG edges carry explicit types: *causes*, *contradicts*, *subsumes*. Two retrieved chunks tell the LLM about a relationship, but the LLM re-derives its type every time under context pressure, with no cross-session consistency guarantee. The KG is more consistent; the LLM is more flexible.

**Global graph algorithms.** PageRank, community detection, and shortest-path across an entire corpus are not possible in a context window. If your product question is "what is the most central concept linking all 50,000 of our contracts?" you need a graph. For most enterprise document AI this question does not arise.

**Completeness guarantees.** Graph traversal is exhaustive — it finds every edge matching a predicate. LLM-driven retrieval is probabilistic — it stops when it judges it has enough. For compliance audits and drug-interaction safety, a missed edge has real consequences; probabilistic retrieval is insufficient.

**Ontology and inheritance.** LLMs have internalized a large amount of general ontological structure, and domain-specific hierarchies can be injected into context. This is roughly equivalent to a KG for most enterprise verticals.

## How Memory and Adaptive Evolution Close the Gap

The **memory layer** eliminates the most wasteful pattern in iterative retrieval: re-fetching the same chunks within or across sessions. Episodic memory records successful retrieval paths — "for indemnification exposure, the winning path was: vector search → read clause text → structured_lookup for counterparty history." Replaying this path on the next similar query skips the exploratory hops. This is usage-driven approximation of KG traversal, built from evidence rather than hand-crafted ontology.

The **adaptive evolution engine** materializes frequently-needed cross-references into the structured store. When the gap detector finds "users always issue a second structured_lookup for litigation history after querying contract terms," it proposes adding a foreign-key cross-reference between the two collections. That cross-reference *is* a KG edge — but it is derived from real queries, not upfront domain modeling. This inverts the KG construction problem: instead of building the graph an expert thinks users will need, the system builds exactly the graph users actually traverse.

## When a KG Is Still the Right Choice

CogBase's LLM-driven architecture covers roughly 80% of what an explicit KG provides for enterprise document AI, at significantly lower engineering, model and maintenance cost. The remaining cases where a KG remains superior:

- **Completeness-critical applications** — drug interaction safety, AML fraud detection, regulatory compliance where a missed edge has legal or safety consequences
- **Very long chain traversal** — 6+ hops at scale across hundreds of thousands of nodes
- **Global graph algorithms** — centrality, community detection, shortest path over a full corpus

These are real but narrow. The trend line runs against heavyweight KGs: stronger LLM reasoning, longer context windows, and better tool-use training each expand the coverage of the retrieval-based approach. The domains where KGs hold ground share one property — exhaustive, provably complete traversal where missing an edge matters. If your application sits there, build the graph. If you are building enterprise document AI for legal, finance, insurance, or research, the LLM-driven architecture wins on almost every practical dimension.
