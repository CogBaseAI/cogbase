# Legal Contract Analyst Pack

Pre-configured CogBase pack for ingesting and querying large volumes of legal contracts. Each document is processed by an LLM that extracts a single structured `ContractRecord` — covering contract basics, common clause text, and flexible fields for terms and conditions that vary by contract type. The full query engine is then available for structured lookup, semantic search, hybrid reasoning, and grounded report generation.

## Quick start

```python
import openai
from packs.legal.contract_analyst import LegalContractApp, IngestResult
from cogbase.core.models import Document
from cogbase.stores.structured.sqlite import SQLiteStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from cogbase.embeddings import SentenceTransformersEmbedding
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker

client = openai.AsyncOpenAI(api_key="...")

app = LegalContractApp(
    client=client,
    model="claude-sonnet-4-6",
    structured_store=SQLiteStructuredStore("contracts.db"),
    vector_store=FAISSVectorStore(dim=384),
    embedder=SentenceTransformersEmbedding(),
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
        print(f"{r.doc_id}: {r.records_extracted} record extracted")
    else:
        print(f"{r.doc_id}: failed — {r.error}")

result = await app.query("which contracts expire before 2026-01-01?")
print(result.answer)
```

## Structured-only mode

Omit `vector_store`, `embedder`, and `chunker` to run without semantic search. Contract extraction still works, but query routing is limited to Pattern A and Pattern D.

```python
app = LegalContractApp(
    client=client,
    model="claude-sonnet-4-6",
    structured_store=SQLiteStructuredStore("contracts.db"),
)
```

## Extracted contract record

Each ingested document produces exactly one `ContractRecord`. The LLM is instructed to copy clause text verbatim — not paraphrase — so the stored text can serve as a citation.

### Contract basics

| Field | Type | Description |
|-------|------|-------------|
| `contract_id` | `str` | Stable unique ID: `{doc_id}_{uuid}` |
| `doc_id` | `str` | Source document identifier |
| `contract_type` | `str \| None` | Category: `"NDA"`, `"SaaS"`, `"employment"`, `"vendor"`, `"lease"`, etc. |
| `purpose` | `str \| None` | One sentence describing what the contract is for |
| `effective_date` | `str \| None` | Contract start date in `YYYY-MM-DD` format |
| `expiry_date` | `str \| None` | Contract end/expiry date in `YYYY-MM-DD` format |
| `parties` | `list[Party]` | All named parties in the contract. Each party includes `name`, optional `role`, and optional `jurisdiction`. `[]` if none. |
| `contract_value` | `float \| None` | Total monetary value in `currency` units |
| `currency` | `str \| None` | ISO 4217 currency code (e.g. `"USD"`) |

### Common clause text

Verbatim text copied from the contract. `null` when the clause is absent.

| Field | Description |
|-------|-------------|
| `termination` | Termination rights and procedures |
| `liability` | Limitation of liability |
| `governing_law` | Governing law and jurisdiction |
| `confidentiality` | Confidentiality and non-disclosure obligations |
| `indemnification` | Indemnification obligations |
| `dispute_resolution` | Arbitration, mediation, or litigation provisions |

### Structured payment terms

`payment_terms` is a nested object, not a plain string.

| Field | Type | Description |
|-------|------|-------------|
| `payment_terms.schedule` | `str \| None` | Schedule such as `net-30`, `monthly`, `upfront`, or `milestone-based` |
| `payment_terms.due_date` | `str \| None` | Due date in `YYYY-MM-DD` format when explicitly stated |
| `payment_terms.late_penalty` | `str \| None` | Late fee / interest language (verbatim) |
| `payment_terms.verbatim` | `str \| None` | Verbatim payment terms clause text |

### Clause-level numeric

| Field | Type | Description |
|-------|------|-------------|
| `notice_period_days` | `int \| None` | Days of notice required for termination |
| `liability_cap` | `float \| None` | Liability cap amount in `currency` units |

### Flexible fields

| Field | Type | Description |
|-------|------|-------------|
| `key_terms` | `list[str]` | Significant defined terms, unusual provisions, or contract-type-specific clauses not covered by the named fields. `[]` if none. |
| `special_conditions` | `list[str]` | Verbatim text of conditions precedent, carve-outs, custom provisions, or anything unusual. `[]` if none. |

## Customising the schema

Different deployments may need a different set of fields. `build_contracts_schema` lets you add or remove fields without touching `CONTRACTS_SCHEMA`.

```python
from cogbase.stores.schema import FieldSchema, FieldType
from packs.legal.contract_analyst.schema import build_contracts_schema

# Remove fields your organisation does not need
schema = build_contracts_schema(exclude={"indemnification", "dispute_resolution"})

# Add a company-specific field
schema = build_contracts_schema(
    extra_fields={"risk_score": FieldSchema(type=FieldType.FLOAT, nullable=True)}
)

# Both at once
schema = build_contracts_schema(
    extra_fields={"jurisdiction": FieldSchema(type=FieldType.STRING, nullable=True, index=True)},
    exclude={"dispute_resolution"},
)
```

`contract_id` and `doc_id` are core fields and cannot be excluded.

Pass the custom schema to a `StructuredCollection` directly if you need to override it at the `Application` level. For most use cases, `LegalContractApp` uses `CONTRACTS_SCHEMA` directly.

## Query patterns

Queries are automatically routed to the correct retrieval strategy. No configuration needed.

### Pattern A — Structured lookup (no LLM)

Answered directly from the structured store. Fast, no generation cost.

```python
result = await app.query("which contracts expire before 2026-01-01?")
result = await app.query("list all contracts governed by New York law")
result = await app.query("show all contracts where Acme Corp is listed in parties")
result = await app.query("which contracts have a liability cap above 1 million?")
result = await app.query("how many NDA contracts are in the portfolio?")
```

### Pattern B — Semantic search

Open-ended questions answered from raw contract text via vector similarity. Requires the vector store to be configured.

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
result = await app.query("which contracts have unusually long notice periods compared to the others?")
result = await app.query("which NDAs have shorter confidentiality periods than our standard template?")
result = await app.query("are there contracts where the liability cap seems low for the contract value?")
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
| `structured_store` | `StructuredStoreBase` | yes | Persistent store for extracted contract records |
| `vector_store` | `VectorStoreBase` | no | Vector store for raw contract text |
| `embedder` | `EmbeddingBase` | no | Embedder for contract text chunks |
| `chunker` | `ChunkerBase` | no | Chunker for splitting contract text |
| `name` | `str` | no | Logical application name (default: `"legal"`) |
| `extractor_max_tokens` | `int` | no | Max tokens for contract extraction (default: `4096`) |
| `generator_max_tokens` | `int` | no | Max tokens for answer generation (default: `1024`) |
| `retriever_top_k` | `int` | no | Vector search neighbours to retrieve (default: `10`) |

`vector_store`, `embedder`, and `chunker` must all be provided together or all omitted.

### `ingest_many(contracts, *, concurrency=5)`

Ingests a list of documents concurrently. Never raises on individual failures — errors are captured per document.

```python
results: list[IngestResult] = await app.ingest_many(contracts, concurrency=5)
```

Accepts `Document` objects.

### `IngestResult`

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Document identifier |
| `success` | `bool` | `True` when ingestion completed without error |
| `records_extracted` | `int` | `1` on success, `0` when text was blank or LLM output was unparseable |
| `error` | `Exception \| None` | The exception raised, or `None` on success |

## Known limitations

- **Template comparison** — deviation from a standard template requires a reference document concept that does not yet exist.
- **Date format** — dates are stored as `YYYY-MM-DD` strings. The extractor normalises common formats, but ambiguous dates (e.g. "the last day of the fiscal year") will be `null`.
- **One record per document** — the extractor produces a single `ContractRecord` per ingested document. Consolidated agreements or multi-part contracts should be split into individual documents before ingestion.
