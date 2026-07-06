# Contract Analyst Demo

Ask natural-language questions across a portfolio of legal contracts: find which agreements expire before a given date, compare termination rights across vendors, surface contracts with unusual liability caps, or retrieve the verbatim clause text for any obligation. Structured lookups (e.g. "list all contracts governed by New York law") return exact records; open-ended questions stream a synthesized answer with citations.

The demo ships with a portfolio of **thirty contract fixtures** for a single fictional company (Meridian Analytics Inc.), spanning the document types an SMB or enterprise legal team actually juggles — SaaS subscriptions, NDAs, employment and contractor agreements, a separation/release, office and equipment leases, master consulting/services agreements and a SOW, a master services agreement, vendor supply and marketing agreements, a standalone DPA, a perpetual license, a reseller agreement, an evaluation agreement, a master purchase agreement, and three amendments. They live as plain text in `contracts.py`, and the demo also accepts any plain-text contract you provide.

The corpus is engineered with **planted cross-document tensions** so the hero queries below — the cross-document reasoning that plain chunk-and-retrieve RAG cannot do — land hard: two vendors each appear on agreements whose payment terms conflict, three amendments flip clauses in their originals, and there are deliberate liability-cap, notice-period, and expiry outliers. See the module docstring in `contracts.py` for the full catalogue.

On ingest, each fixture is rendered to a `.docx` on the fly and uploaded (parsed to markdown server-side) — no Word files are committed to the repo. Each ingested document produces one structured record (parties, dates, clause text, payment terms) plus a searchable vector index of its full text.

## Quick start

```bash
# 1. Start the API server with Docker — no build required (see ../../server/README.md)
./server/docker_hub_demo.sh pull
./server/docker_hub_demo.sh run

# 2. Run the demo (from repo root)
python examples/contract_analyst_demo/demo.py
```

The API server runs at `http://localhost:8000`. After the container starts, configure your LLM and embedding provider (including API key) via the UI Settings tab. See [`server/README.md`](../../server/README.md) for details, including how to pull a specific version, persist data, or serve on a different port.

## Interactive commands

| Command | Description |
|---------|-------------|
| `list` | List all applications |
| `create` | Create the contract-analyst application |
| `delete <name>` | Delete an application by name (with confirmation) |
| `/ingest_demo_contracts` | Ingest the built-in contract fixtures (30 agreements across types, incl. 3 amendments) |
| `ingest <path>` | Ingest a plain-text contract file from disk |
| `list collections` | List all structured collections for the application |
| `query structured` | Query the default `contracts` collection (all records) |
| `query structured <name>` | Query a named structured collection (all records) |
| `reset` | Delete the application and all data |
| `q` / `quit` / `exit` | Exit |

Any other input is sent as a query to the running application. Answers stream back as SSE tokens. Passthrough structured results (e.g. contract lookups) are printed as JSON.

## What the demo creates

On first run the demo uploads a ZIP bundle to `POST /applications` that configures:

- **Pipeline steps**:
  1. `chunk-embed-upsert` → `document_chunks` vector collection
  2. `extract-structured` → `contracts` structured collection

The bundle contains three files:

| File | Purpose |
|------|---------|
| `contracts_record_schema.json` | Record schema — what is stored (`ContractExtractionRecord`, includes `doc_id`) |
| `contracts_extraction_schema.json` | Extraction schema — what the LLM extracts (`ContractExtraction`, no `doc_id`) |
| `contracts_prompt.txt` | LLM system prompt prefix |

The extraction schema and record schema are separate: `ContractExtraction` defines the fields the LLM fills in; `ContractExtractionRecord` extends it with `doc_id`, which the pipeline injects automatically. The collection is declared with `schema: contracts_record_schema.json` and the extractor with `extraction_schema: contracts_extraction_schema.json`.

If the application already exists, the demo reuses it.

## Stored contract record

Each ingested document produces exactly one record in the `contracts` collection. The LLM is instructed to copy clause text verbatim — not paraphrase — so stored text can serve as a citation.

### Identity

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `doc_id` | `str` | Injected by pipeline | Source document identifier — not extracted by the LLM |

### Contract basics

| Field | Type | Description |
|-------|------|-------------|
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

Cross-document hero queries (exploit the planted tensions):

```
> we have multiple agreements with TechVault Solutions — do their payment terms conflict?
> which agreements contain non-compete or non-solicitation covenants, and how long do they last?
> which agreements have unlimited liability or no stated liability cap?
> list every agreement that expires before March 1, 2026, regardless of contract type
> merge the amendment into the original saas-001 contract and give me the updated document
```

## Project structure

```text
contract_analyst_demo/
├── README.md
├── demo.py               # interactive demo script
├── schema.py             # ContractExtraction, ContractExtractionRecord, Party, PaymentTerms
└── contracts.py          # built-in contract fixtures (30 agreements, rendered to .docx at ingest)
```

## Known limitations

- **Date format** — dates are stored as `YYYY-MM-DD` strings. Ambiguous dates (e.g. "the last day of the fiscal year") will be `null`.
- **One record per document** — the extractor produces a single record per ingested document. Consolidated or multi-part contracts should be split before ingestion.
