# Legal Contract Analyst Pack

Pre-configured CogBase pack for ingesting and querying large volumes of legal contracts. Extracts typed clauses via LLM, stores them in a structured store, and exposes the full query engine for lookup, semantic search, hybrid reasoning, and grounded report generation.

## Quick start

```python
import openai
from packs.legal import LegalContractApp, IngestResult
from cogbase.core.models import Document
from cogbase.stores.structured.sqlite import SQLiteStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from cogbase.pipeline.ingestion.embedder import SentenceTransformersEmbedder
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker

client = openai.AsyncOpenAI(api_key="...")

app = LegalContractApp(
    client=client,
    model="claude-sonnet-4-6",
    structured_store=SQLiteStructuredStore("contracts.db"),
    vector_store=FAISSVectorStore(dim=384),
    embedder=SentenceTransformersEmbedder(),
    chunker=FixedSizeChunker(chunk_size=512, overlap=64),
)

await app.setup()

# Ingest a batch of contracts
results = await app.ingest_many([
    Document(doc_id="vendor-001", text=vendor_text),
    Document(doc_id="nda-002",    text=nda_text),
    Document(doc_id="lease-003",  text=lease_text),
], concurrency=5)

for r in results:
    if r.success:
        print(f"{r.doc_id}: {r.clauses_extracted} clauses extracted")
    else:
        print(f"{r.doc_id}: failed — {r.error}")

result = await app.query("what are the termination clauses?")
print(result.answer)
```

## Structured-only mode

Omit `vector_store`, `embedder`, and `chunker` to run without semantic search. Clause extraction and Pattern A structured lookups still work; Pattern B queries return empty results.

```python
app = LegalContractApp(
    client=client,
    model="claude-sonnet-4-6",
    structured_store=SQLiteStructuredStore("contracts.db"),
)
```

## Extracted clause types

The extractor identifies the following clause categories:

| Type | Description |
|------|-------------|
| `payment` | Payment terms, schedules, and amounts |
| `termination` | Termination rights and conditions |
| `liability` | Liability caps and exclusions |
| `notice` | Notice periods and delivery requirements |
| `governing_law` | Governing law and jurisdiction |
| `confidentiality` | Confidentiality and non-disclosure obligations |
| `indemnification` | Indemnification obligations |
| `dispute_resolution` | Arbitration, mediation, and litigation provisions |
| `other` | Any other significant clause |

Each extracted clause carries: `clause_id`, `doc_id`, `type`, `text` (verbatim), `page`, `confidence`.

## Query patterns

Queries are automatically routed to the correct retrieval strategy. No configuration needed.

### Pattern A — Structured lookup (no LLM)

Answered directly from the structured store. Fast, no generation cost.

```python
result = await app.query("which contracts have a termination-for-convenience clause?")
result = await app.query("list all contracts governed by New York law")
result = await app.query("how many contracts contain a limitation of liability clause?")
result = await app.query("show me all indemnification clauses with high confidence")
```

### Pattern B — Semantic search

Open-ended questions answered from raw contract text via vector similarity.

```python
result = await app.query("find passages about data breach notification obligations")
result = await app.query("which contracts mention GDPR or data residency requirements?")
result = await app.query("find language about audit rights")
result = await app.query("are there any clauses that restrict assignment to competitors?")
```

### Pattern C — Hybrid reasoning

Retrieves from both stores and reasons across the combined results.

```python
result = await app.query("do any contracts contradict each other on payment terms with Vendor X?")
result = await app.query("which contracts have unusually long cure periods compared to the others?")
result = await app.query("which NDAs have shorter confidentiality periods than our standard template?")
result = await app.query("are there any contracts where notice periods differ between the parties?")
```

### Pattern D — Grounded report

Generates a structured deliverable. The result separates `findings` from `supporting_quotes` — every quote is verbatim from the source contracts.

```python
result = await app.query("summarise all termination rights across the vendor portfolio")
result = await app.query("which contracts are most risky? explain with supporting quotes")
result = await app.query("produce a comparison of governing law and dispute resolution clauses")
result = await app.query("what renewal obligations do we have in the next 90 days?")

print(result.findings)
print(result.supporting_quotes)  # list[str] — verbatim excerpts
```

## API reference

### `LegalContractApp`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `client` | `AsyncOpenAI` | yes | OpenAI-compatible async client |
| `model` | `str` | yes | Model name for extraction, routing, and generation |
| `structured_store` | `StructuredStoreBase` | yes | Persistent store for extracted clauses |
| `vector_store` | `VectorStoreBase` | no | Vector store for raw contract text |
| `embedder` | `EmbedderBase` | no | Embedder for contract text chunks |
| `chunker` | `ChunkerBase` | no | Chunker for splitting contract text |
| `extractor_max_tokens` | `int` | no | Max tokens for clause extraction (default: 4096) |
| `generator_max_tokens` | `int` | no | Max tokens for answer generation (default: 1024) |
| `retriever_top_k` | `int` | no | Vector search neighbours to retrieve (default: 10) |

`vector_store`, `embedder`, and `chunker` must all be provided together or all omitted.

### `ingest_many(contracts, *, concurrency=5)`

Ingests a list of documents concurrently. Never raises on individual failures — errors are captured per document.

```python
results: list[IngestResult] = await app.ingest_many(contracts, concurrency=5)
```

Accepts `Document` objects or `(text, doc_id)` tuples.

### `IngestResult`

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Document identifier |
| `success` | `bool` | `True` when ingestion completed without error |
| `clauses_extracted` | `int` | Number of clauses written to the structured store |
| `error` | `Exception \| None` | The exception raised, or `None` on success |

## Known limitations

The current `CLAUSES_SCHEMA` handles clause types well but is missing fields that drive many high-value Pattern A queries:

- **Dates** — expiry dates, effective dates, and notice periods need typed fields for date-range lookups ("contracts expiring before December 31, 2025").
- **Party names** — "all contracts with Acme Corp" requires a structured field; `doc_id` is the only identifier today.
- **Contract value** — numeric comparisons ("liability cap lower than contract value") need amount fields.
- **Contradiction detection** — cross-contract inconsistency detection is planned in the CogBase architecture but not yet implemented in this pack.
- **Template comparison** — deviation from a standard template requires a reference document concept that does not yet exist.
