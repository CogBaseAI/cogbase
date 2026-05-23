# Legal Case Preparation Demo

Upload an entire case bundle — correspondence, contracts, invoices, witness
statements, expert reports, court orders, pleadings, and any disclosed
evidence — and let the system build the working artefacts a lawyer needs at
the start of a dispute:

1. **Document classification and inventory** — every file is tagged with a
   type, a short summary, a relevance tag, the authors and recipients, and a
   date range, so any document can be located in seconds.
2. **Chronological timeline of events** — every dated event, communication,
   obligation, payment, notice, delivery, or action is pulled out, attached
   to its source quote, and merged into a single unified timeline.
3. **Cast of characters** — every named individual, company, organisation,
   and government body is extracted with its role at the time, the documents
   it appears in, and the entities it is connected to.
4. **Fact matrix per party** — every factual assertion is captured with the
   party asserting it, the source quote, and an issue tag, so all facts
   bearing on the same disputed point can be lined up across documents.
5. **Contradiction detection** — the contradiction-detection workflow reads
   every fact for an issue and surfaces incompatible claims as side-by-side
   pairs, ranked by likely legal significance.
6. **Evidence gap identification** — the evidence-gap workflow flags any
   assertion that no other document corroborates, with the potential impact
   and a suggested next step (request further disclosure, commission an
   expert report, take an additional statement).
7. **Structured reference table** — dates and deadlines, monetary amounts,
   payments, obligations with their clause references, notices served, and
   alleged breaches are pulled into a clean cross-document table.

After ingestion you can also ask free-form questions across the case bundle —
the query runner has both the structured collections and the per-document
chunks available for retrieval.

## Quick start

```bash
# 1. Start the API server
uvicorn api.main:app --reload --log-level info

# 2. Run the demo from the repo root
python examples/legal_case_prep_demo/demo.py
```

Requires `OPENAI_API_KEY` in a `.env` file at the repo root or in the
environment. Set `COGBASE_API_URL` to override the default
`http://localhost:8000`.

The workflow commands require persistent store backends (SQLite + FAISS).
Configure `cogbase_system.yaml` with `structured_store.type=sqlite` and
`vector_store.type=faiss`, or set `COGBASE_CONFIG` to point to your system
config.

## Interactive commands

| Command | Description |
|---------|-------------|
| `/ingest_demo_case` | Ingest the built-in nine-document case bundle |
| `/ingest_case <path>` | Ingest a single case document from disk |
| `/inventory` | Show the document inventory |
| `/timeline [<issue>]` | Show the chronological timeline of events |
| `/cast` | Show the cast of characters |
| `/facts [<issue>]` | Show extracted facts |
| `/reference_table [<kind>]` | Show the structured data reference table |
| `/detect_contradictions <issue>` | Run the contradiction-detection workflow |
| `/find_gaps <issue>` | Run the evidence-gap workflow |
| `/contradictions [<issue>]` | List saved contradictions |
| `/gaps [<issue>]` | List saved evidence gaps |
| `/q`, `/quit`, `/exit` | Exit |

Any other input is sent as a natural-language query to the `legal-case-prep`
app.

## Demo workflow

```text
1. /ingest_demo_case
2. /inventory
3. /timeline
4. /cast
5. /facts delivery-dispute
6. /detect_contradictions delivery-dispute
7. /find_gaps defect-allegation
8. /reference_table monetary_amount
```

## Example queries

```text
> which documents discuss the delivery on 14 March 2025?
> who is John Reid and which documents mention him?
> what does Beacon allege about the condition of the valves?
> summarise the case against Acme based on the witness statement
```

## What the demo creates

The demo creates one CogBase application named `legal-case-prep`.

### Vector collections

| Collection | Purpose |
|------------|---------|
| `case_chunks` | Passage-level chunks of every ingested document; used for detailed retrieval and citations |
| `case_summaries` | One short summary per document; used for broad "which documents mention X" questions |

### Structured collections

| Collection | Primary key | Source | Purpose |
|------------|-------------|--------|---------|
| `case_documents` | `doc_id` | Pipeline extraction (one) | Document inventory: type, title, summary, relevance, authors, dates |
| `timeline_events` | `event_id` | Pipeline extraction (many) | Every dated event, communication, obligation, action |
| `entities` | `entity_id` | Pipeline extraction (many) | Every named person, company, organisation, government body |
| `facts` | `fact_id` | Pipeline extraction (many) | Factual assertions with party attribution and source quote |
| `structured_data` | `item_id` | Pipeline extraction (many) | Dates, amounts, payments, obligations, notices, alleged breaches |
| `contradictions` | `contradiction_id` | Workflow output | Pairs of incompatible factual claims, ranked by significance |
| `evidence_gaps` | `gap_id` | Workflow output | Uncorroborated assertions with potential impact and suggested action |

## Pipeline shape

A single `case-document` pipeline handles every ingested document, regardless
of format (PDF, email, Word, scan). The same six steps run in declaration
order per document:

```text
chunk-embed-upsert    → case_chunks               (passage retrieval)
extract-structured    → case_documents            (record_mode=one)
extract-structured    → timeline_events           (record_mode=many)
extract-structured    → entities                  (record_mode=many)
extract-structured    → facts                     (record_mode=many)
extract-structured    → structured_data           (record_mode=many)
document-embed-upsert → case_summaries            (one summary per document)
```

Each extraction step uses its own JSON Schema and system prompt. Identity
fields (`doc_id`, `event_id`, `entity_id`, `fact_id`, `item_id`) are injected
by the extractor and declared on the record schemas; the LLM only fills the
substantive fields.

## Workflows

### `detect-contradictions`

Input: `{"issue": "delivery-dispute"}`

```text
load_facts (structured-query, filtered by issue)
  → judge (llm-structured)
      input  = { issue, facts }
      schema = ContradictionList
  → save_each (foreach output.contradictions)
      → save_one (structured-save → contradictions)
```

The judge is grounded only in the supplied facts and the verbatim source
quotes. Each contradiction records both fact IDs, both source quotes, both
asserting parties, an explanation, a significance ranking, and reasoning.

### `identify-evidence-gaps`

Input: `{"issue": "defect-allegation"}`

```text
load_facts (structured-query, filtered by issue)
load_structured_data (structured-query, filtered by issue)
  → judge (llm-structured)
      input  = { issue, facts, structured_data }
      schema = EvidenceGapList
  → save_each (foreach output.gaps)
      → save_one (structured-save → evidence_gaps)
```

The judge treats a single party's restatement of the same assertion as
non-corroborative; only independent contemporaneous records or assertions
from a different party count. Each gap records the missing corroboration,
the potential impact, and a concrete next step.

Both workflows are idempotent: the primary keys (`contradiction_id`,
`gap_id`) are derived from the issue and source fact IDs so reruns overwrite
prior findings in place.

## Sample case bundle

The built-in bundle is a fictional commercial dispute between Acme Industrial
Supplies Limited (claimant) and Beacon Manufacturing PLC (defendant) over
the supply of 200 ML-7 industrial valves. Nine documents are included:

| doc_id | Type | Description |
|--------|------|-------------|
| `doc-001` | contract | Supply Agreement signed 1 Feb 2025 |
| `doc-002` | correspondence | Beacon's order confirmation email (8 Feb 2025) |
| `doc-003` | disclosed_evidence | Acme delivery note signed 14 Mar 2025 |
| `doc-004` | witness_statement | Sarah Patel statement of 12 May 2025 |
| `doc-005` | correspondence | Acme MD email demanding payment (25 Mar 2025) |
| `doc-006` | correspondence | Beacon's solicitor letter (5 Apr 2025) |
| `doc-007` | correspondence | Acme's termination notice (10 May 2025) |
| `doc-008` | expert_report | Dr Vance's inspection report (1 Jul 2025) |
| `doc-009` | pleading | Particulars of Claim (30 Jul 2025) |

The bundle is deliberately built so the pipeline produces real findings:

- **delivery-dispute**: Acme's delivery note (200 units accepted on 14 Mar)
  vs Beacon's witness statement (180 units delivered on 16 Mar).
- **defect-allegation**: Beacon claims five visibly damaged units; the
  expert later identifies four defective units.
- **payment-default / breach-notice**: Acme treats the deemed-acceptance
  period as having run; Beacon's solicitor disputes that any payment was
  due.
- Several assertions (e.g. Beacon's photographs of the damage, the 20
  missing units) lack any independent corroboration, surfacing as evidence
  gaps.

## Project structure

```text
legal_case_prep_demo/
├── README.md
├── config.yaml                         # pipeline + two workflows
├── demo.py                             # interactive demo script and ZIP bundle builder
├── schema.py                           # Pydantic extraction and record models
├── case_data.py                        # sample nine-document case bundle
├── case_document_prompt.txt
├── timeline_event_prompt.txt
├── entity_prompt.txt
├── fact_prompt.txt
├── structured_data_prompt.txt
├── contradiction_judge_prompt.txt
└── evidence_gap_judge_prompt.txt
```

JSON Schema files for both the extraction and record schemas are generated
from the Pydantic models at bundle-build time in `demo.py` and written into
the ZIP sent to `POST /applications`.

## Design notes

**One pipeline, many extractions.** Every document goes through the same
pipeline regardless of format. Five `extract-structured` steps run per
document — one returns a single record (the inventory), the other four
return many records (events, entities, facts, structured data). The pipeline
runs all steps in order; partial failures of one step do not block the
others.

**Issue tagging as the join key.** The fact matrix, contradiction detection,
and evidence-gap identification all rely on the LLM applying a consistent
`issue` tag across documents. The prompts emphasise reusing the same
kebab-case slugs (e.g. `delivery-dispute`, `breach-notice`,
`liability-allocation`) so cross-document analyses work.

**Workflows, not skills.** Contradiction detection and evidence-gap
identification have a fixed execution graph — load → judge → save per
record — so they live in YAML workflows rather than agentic skills. The LLM
only judges; it does not decide which steps to run.

**LLM groundedness.** Both judges are constrained to ground every finding in
the supplied facts and their verbatim quotes. The judges may not invent
contradicting positions or corroborating sources that are not in the input.

## Known limitations

- Extraction quality on scanned or OCR-derived documents depends on the
  upstream parser used by the API server's `upload_documents` endpoint.
- The contradiction and gap judges work per-issue; very broad issue tags
  can blow past the LLM context window. Use narrower tags for very large
  case bundles.
- The system is preparation-assistive and produces draft findings only; it
  is not legal advice and does not replace counsel review.
