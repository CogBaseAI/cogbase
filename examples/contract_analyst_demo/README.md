# Contract Analyst Demo

A CogBase demo that ingests and queries legal contracts through the REST API. Each document is processed by an LLM that extracts a structured `ContractExtraction` record — covering contract basics, common clause text, and flexible fields for terms and conditions that vary by contract type.

## Quick start

```bash
# 1. Start the API server
uvicorn api.main:app --reload

# 2. Run the demo (from repo root)
python examples/contract_analyst_demo/demo.py
```

Requires `OPENAI_API_KEY` in a `.env` file at the repo root (or in the environment). Set `COGBASE_API_URL` to override the default `http://localhost:8000`.

## Interactive commands

| Command | Description |
|---------|-------------|
| `list` | List all applications |
| `create` | Create the contract-analyst application |
| `delete <name>` | Delete an application by name (with confirmation) |
| `ingest saas` | Ingest the 5 built-in SaaS contract fixtures |
| `ingest <path>` | Ingest a plain-text contract file from disk |
| `list collections` | List all structured collections for the application |
| `query structured` | Query the default `contracts` collection (all records) |
| `query structured <name>` | Query a named structured collection (all records) |
| `reset` | Delete the application and all data |
| `q` / `quit` / `exit` | Exit |

Any other input is sent as a query to the running application. Answers stream back as SSE tokens. Passthrough structured results (e.g. contract lookups) are printed as JSON.

## What the demo creates

On first run the demo uploads a ZIP bundle to `POST /applications` that configures:

- **LLM**: `gpt-5.4-mini` via OpenAI
- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **Pipeline steps**:
  1. `chunk-embed-upsert` → `document_chunks` collection
  2. `extract-structured` → `contracts` collection

The bundle includes the JSON schema for `ContractExtraction` and the extraction system prompt. If the application already exists, the demo reuses it.

## Extracted contract record

Each ingested document produces exactly one `ContractExtraction`. The LLM is instructed to copy clause text verbatim — not paraphrase — so stored text can serve as a citation.

### Contract basics

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Source document identifier (injected by the pipeline) |
| `contract_type` | `str \| None` | Category: `"NDA"`, `"SaaS"`, `"employment"`, `"vendor"`, `"lease"`, etc. |
| `purpose` | `str \| None` | One sentence describing what the contract is for |
| `effective_date` | `str \| None` | Contract start date in `YYYY-MM-DD` format |
| `expiry_date` | `str \| None` | Contract end/expiry date in `YYYY-MM-DD` format |
| `parties` | `list[Party]` | All named parties. Each includes `name`, optional `role`, optional `jurisdiction`. `[]` if none. |
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
| `key_terms` | `list[str]` | Significant defined terms, unusual provisions, or contract-type-specific clauses. `[]` if none. |
| `special_conditions` | `list[str]` | Verbatim text of conditions precedent, carve-outs, or custom provisions. `[]` if none. |

## Example queries

```
> which contracts expire before 2026-01-01?
> list all contracts governed by New York law
> find passages about data breach notification obligations
> which contracts have unusually long notice periods?
> summarise all termination rights across the vendor portfolio
```

## Known limitations

- **Date format** — dates are stored as `YYYY-MM-DD` strings. Ambiguous dates (e.g. "the last day of the fiscal year") will be `null`.
- **One record per document** — the extractor produces a single record per ingested document. Consolidated or multi-part contracts should be split before ingestion.
