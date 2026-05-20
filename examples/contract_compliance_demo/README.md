# Contract Compliance Demo

Upload a set of company rules and incoming contracts, then let the system review every clause automatically: each clause is checked against your policy library, rated by severity, and stored with the exact rule passages that justify the finding. A non-compliant clause also gets a suggested redline. Results appear as they stream in — no waiting for the full document to finish.

After a review is complete you can query across findings: list all high-severity issues, ask which contracts violate a specific rule, or pull the full compliance report for any contract. The demo ships with built-in rule fixtures and sample contracts so you can run the full workflow end-to-end without any external documents.

## Quick start

```bash
# 1. Start the API server
uvicorn api.main:app --reload

# 2. Run the demo from the repo root
python examples/contract_compliance_demo/demo.py
```

Requires `OPENAI_API_KEY` in a `.env` file at the repo root or in the
environment. Set `COGBASE_API_URL` to override the default
`http://localhost:8000`.

The `check`, `report`, and `alerts` commands require persistent store backends
(SQLite + FAISS). Configure `cogbase_system.yaml` with
`structured_store.type=sqlite` and `vector_store.type=faiss`, or set
`COGBASE_CONFIG` to point to your system config.

## Interactive commands

| Command | Description |
|---------|-------------|
| `create` | Create the `contract-compliance` application |
| `ingest rules` | Ingest the built-in company rules fixtures |
| `ingest rules <path>` | Ingest a company rules document from disk |
| `ingest contracts` | Ingest the built-in contract fixtures |
| `ingest contract <path>` | Ingest a contract file from disk |
| `check <doc_id>` | Run the compliance workflow for one contract |
| `report <doc_id>` | Print the stored compliance report for one contract |
| `alerts` | List high and critical compliance findings |
| `list collections` | List collections for the application |
| `query structured <name>` | Dump a structured collection |
| `reset` | Delete the application and all demo data |
| `q` / `quit` / `exit` | Exit |

Any other input is sent as a natural-language query to the
`contract-compliance` app.

## Demo workflow

```text
1. create
2. ingest rules                         -> rule_chunks populated
3. ingest contracts                     -> contract_chunks, contract_clauses,
                                           contract_metadata populated
4. check contract-001                   -> clause_compliance_findings populated
5. report contract-001
6. alerts
```

Example questions after ingestion and review:

```text
> show all non-compliant clauses for contract-001
> which clauses have high-severity findings?
> what company rule does the liability clause violate?
> summarize the compliance report for contract-001
> are there any contracts with no compliance findings?
```

## What the demo creates

The demo creates one CogBase application named `contract-compliance`.

### Vector collections

| Collection | Routed documents | Purpose |
|------------|------------------|---------|
| `rule_chunks` | `metadata.doc_type == "rules"` | Searchable company rules, standards, and fallback positions |
| `contract_chunks` | `metadata.doc_type == "contract"` | Searchable contract passages for QA and citation |

### Structured collections

| Collection | Primary key | Source | Purpose |
|------------|-------------|--------|---------|
| `contract_metadata` | `doc_id` | Pipeline extraction | One record of key contract facts per contract |
| `contract_clauses` | `clause_id` | Pipeline extraction | One record per extracted clause |
| `clause_compliance_findings` | `clause_id` | Workflow output | One compliance finding per reviewed clause |

## Schema design

Each structured collection uses two schemas:

- **Extraction schema** — the fields the LLM is asked to extract (no identity fields).
- **Record schema** — what is stored in the collection (extraction fields + identity fields such as `doc_id` and `clause_id`, injected by the pipeline).

### `contract_metadata`

Extraction schema: `ContractMetadata` — the LLM extracts these fields.
Record schema: `ContractMetadataRecord` — adds `doc_id` (primary key).

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Source contract document ID (injected, not extracted) |
| `contract_type` | `str \| null` | Contract category |
| `parties` | `list[Party]` | Named parties and their roles |
| `effective_date` | `str \| null` | Start date in `YYYY-MM-DD` format |
| `expiry_date` | `str \| null` | End date in `YYYY-MM-DD` format |
| `contract_value` | `float \| null` | Total monetary value when explicitly stated |
| `currency` | `str \| null` | ISO 4217 currency code |
| `governing_law` | `str \| null` | Governing law jurisdiction |
| `termination_notice_days` | `int \| null` | Notice period in days for termination |

### `contract_clauses`

Extraction schema: `ContractClause` — the LLM extracts one item per clause.
Record schema: `ContractClauseRecord` — adds `clause_id` (primary key, format `{doc_id}__{i:04d}`) and `doc_id`.

| Field | Type | Description |
|-------|------|-------------|
| `clause_id` | `str` | Stable per-clause identifier (injected, not extracted) |
| `doc_id` | `str` | Source contract document ID (injected, not extracted) |
| `clause_type` | `str \| null` | Clause category: `liability`, `indemnification`, `termination`, `payment`, `privacy`, `confidentiality`, `ip`, `governing_law`, `other` |
| `text` | `str` | Verbatim clause text |

### `clause_compliance_findings`

Produced by the compliance workflow, not by pipeline extraction. The LLM judge writes directly to this collection; there is no separate extraction schema.

| Field | Type | Description |
|-------|------|-------------|
| `clause_id` | `str` | Primary key — matches the reviewed clause in `contract_clauses` |
| `doc_id` | `str` | Source contract document ID |
| `clause_type` | `str \| null` | Reviewed clause category |
| `status` | `str` | `compliant`, `non_compliant`, `needs_review`, or `not_applicable` |
| `severity` | `str` | `low`, `medium`, `high`, or `critical` |
| `summary` | `str` | Short human-readable finding |
| `contract_clause_text` | `str` | Verbatim reviewed clause text |
| `matched_rule_ids` | `list[str]` | IDs of rule chunks used as evidence |
| `matched_rule_quotes` | `list[str]` | Verbatim excerpts from matched rule chunks |
| `reasoning` | `str` | Explanation grounded in matched rules |
| `recommended_redline` | `str \| null` | Suggested replacement language; null when compliant |
| `confidence` | `float` | Judge confidence from 0.0 to 1.0 |

## Routed ingestion

Rules documents and contract documents share one app but take different paths
through the ingestion pipeline. The demo relies on document metadata to route
pipeline steps deterministically.

Rules are ingested with `metadata.doc_type = "rules"`; contracts with
`metadata.doc_type = "contract"`. The pipeline shape:

```yaml
pipeline:
  steps:
    - tool: chunk-embed-upsert
      collection: rule_chunks
      when:
        metadata:
          doc_type: rules

    - tool: chunk-embed-upsert
      collection: contract_chunks
      when:
        metadata:
          doc_type: contract

    - tool: extract-structured
      collection: contract_metadata
      extractor:
        type: llm
        extraction_schema: contract_metadata_extraction_schema.json
        prompt: contract_metadata_prompt.txt
      when:
        metadata:
          doc_type: contract

    - tool: extract-structured
      collection: contract_clauses
      extractor:
        type: llm
        extraction_schema: contract_clause_extraction_schema.json
        extract_as_list: true
        list_field: clauses
        item_id_field: clause_id
        prompt: contract_clauses_prompt.txt
      when:
        metadata:
          doc_type: contract
```

## Compliance check workflow

The `check-contract-compliance` workflow is a declarative, config-driven
replacement for a hand-rolled skill. The LLM does not decide which steps to
run; it only judges one contract clause against retrieved company rule
passages.

```yaml
workflows:
  - name: check-contract-compliance
    params_from_collection:
      collection: contract_metadata
      filters:
        doc_id: "{{ doc.doc_id }}"
      params:
        doc_id: "{{ record.doc_id }}"
    steps:
      - id: load_clauses
        tool: structured-query
        collection: contract_clauses
        filters:
          doc_id: "{{ input.doc_id }}"

      - id: review_each_clause
        foreach: "{{ steps.load_clauses.records }}"
        steps:
          - id: retrieve_rules
            tool: vector-search
            collection: rule_chunks
            query: "{{ item.clause_type }}\n{{ item.text }}"
            top_k: 5

          - id: judge
            tool: llm-structured
            prompt: compliance_judge_prompt.txt
            input:
              clause: "{{ item }}"
              rules: "{{ steps.retrieve_rules.chunks }}"
            output_schema: clause_compliance_findings_schema.json

          - id: save_finding
            tool: structured-save
            collection: clause_compliance_findings
            records:
              - "{{ steps.judge.output }}"
```

The workflow is deterministic: every clause for the requested `doc_id` is
reviewed with the same rule collection and `top_k`. Findings are upserted with
`clause_id` as the primary key so rerunning `check <doc_id>` updates prior
findings in place rather than duplicating them.

Results stream as SSE events — findings appear in the terminal as they are
written:

```
POST /applications/contract-compliance/workflows/check-contract-compliance/stream
{"params": {"doc_id": "contract-001"}}
```

## LLM judge rules

The compliance judge (prompt: `compliance_judge_prompt.txt`) receives the clause text and the top matching company rule passages. Constraints:

- Use only the provided rule passages as company policy.
- Return `needs_review` when retrieved rules are insufficient to decide.
- Do not invent company policy or rely on outside legal knowledge.
- Every `non_compliant` finding must cite at least one `matched_rule_quote`.
- Populate `recommended_redline` for non-compliant findings; null otherwise.

## Project structure

```text
contract_compliance_demo/
├── README.md
├── demo.py              # interactive demo script and ZIP bundle builder
├── schema.py            # Pydantic extraction and record models for all three collections
├── rules_data.py        # sample company rule documents
└── contracts_data.py    # sample incoming contracts
```

The pipeline config, prompts, and all schema files are bundled inline in
`demo.py` and written into the ZIP sent to `POST /applications` at startup.

## Design notes

**Why one application?** Rules and contracts are part of the same compliance
workspace. Keeping them in one app avoids cross-app collection access and lets
the query runner see all relevant structured and vector collections.
`metadata.doc_type` controls which pipeline steps apply to each document type.

**Why separate extraction and record schemas?** The extraction schema describes
what the LLM is asked to return; identity fields (`doc_id`, `clause_id`) are
meaningless to the LLM and would add noise to the prompt. The record schema
adds those fields explicitly so the collection definition is self-contained and
auditable. The extractor injects identity values at parse time.

**Why a workflow and not an ingestion step?** Compliance checking reads
previously extracted clause records, searches rule chunks, calls an LLM judge,
and writes findings. That is a cross-collection workflow over persisted records,
not a single-document ingestion step. A declarative workflow makes each stage
inspectable and independently testable without custom Python code.

**Why not a skill?** The compliance check has a fixed execution graph —
load clauses, retrieve rules, judge, save — with no branching that requires LLM
reasoning about which tools to invoke. A workflow expresses this directly in
config without the overhead of an agentic loop.

## Known limitations

- Semantic search may retrieve incomplete policy context for broad or ambiguous clauses.
- The LLM judge is policy-assistive and should not be treated as legal advice.
- Rule versioning is not considered in the demo.
