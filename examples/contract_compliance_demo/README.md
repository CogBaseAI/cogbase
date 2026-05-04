# Contract Compliance Demo

A CogBase demo that checks incoming contracts against a company's internal
rules and standards. The demo uses one application with separate collections
for company rules, contract text, extracted clauses, contract metadata, and
clause-level compliance findings.

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

## Interactive commands

| Command | Description |
|---------|-------------|
| `create` | Create the `contract-compliance` application |
| `ingest rules` | Ingest the built-in company rules fixtures |
| `ingest rules <path>` | Ingest a company rules document from disk |
| `ingest contracts` | Ingest the built-in contract fixtures |
| `ingest contract <path>` | Ingest a contract file from disk |
| `check <doc_id>` | Run deterministic clause-by-clause compliance review |
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

### Chunk collections

| Collection | Routed documents | Purpose |
|------------|------------------|---------|
| `rule_chunks` | `metadata.doc_type == "rules"` | Searchable company rules, standards, playbooks, and fallback positions |
| `contract_chunks` | `metadata.doc_type == "contract"` | Searchable contract passages for QA and citation |

### Structured collections

| Collection | Routed documents | Purpose |
|------------|------------------|---------|
| `contract_clauses` | `metadata.doc_type == "contract"` | One typed record per extracted contract clause |
| `contract_metadata` | `metadata.doc_type == "contract"` | One typed record with key contract facts |
| `clause_compliance_findings` | Written by `check <doc_id>` | One compliance finding per reviewed clause |

## Routed ingestion

Rules documents and contract documents share one app but take different paths
through the ingestion pipeline. The demo relies on document metadata to route
pipeline steps deterministically.

Rules are ingested with:

```json
{
  "doc_id": "rules-001",
  "text": "...",
  "metadata": {
    "doc_type": "rules",
    "source": "vendor_contract_standards.txt",
    "ruleset_id": "company_rules_v1"
  }
}
```

Contracts are ingested with:

```json
{
  "doc_id": "contract-001",
  "text": "...",
  "metadata": {
    "doc_type": "contract",
    "source": "vendor_agreement.txt"
  }
}
```

The intended pipeline shape is:

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
      collection: contract_clauses
      when:
        metadata:
          doc_type: contract

    - tool: extract-structured
      collection: contract_metadata
      when:
        metadata:
          doc_type: contract
```

This keeps the application boundary simple while preserving separate ingestion
lifecycles for rules and contracts.

## Compliance check skill

The `compliance-check` skill is a thin wrapper around a fixed workflow
function. The LLM does not decide which steps to run; it only judges one
contract clause against retrieved company rule passages.

The skill accepts a `doc_id` and runs:

```python
async def check_compliance(doc_id, *, rule_store, clause_store, results_store, llm, embedder):
    clauses = await clause_store.query(
        "contract_clauses",
        filters=[Col("doc_id") == doc_id],
    )

    for clause in clauses:
        query = f"{clause.get('clause_type') or ''}\n{clause['text']}"
        (embedding,) = await embedder.embed([query])

        top_rules = await rule_store.search(
            "rule_chunks",
            query,
            embedding,
            top_k=5,
        )

        finding = await llm_judge(clause, top_rules, llm)
        await results_store.save("clause_compliance_findings", [finding])
```

The workflow is deterministic:

- every extracted clause for the requested `doc_id` is reviewed
- the same rule collection and `top_k` are used for each clause
- findings are upserted with stable IDs
- rerunning `check <doc_id>` updates prior findings instead of duplicating them

## LLM judge rules

The LLM judge receives the clause text and the top matching company rule
passages. It must return validated JSON matching the
`ClauseComplianceFinding` schema.

Judge constraints:

- Use only the provided rule passages as company policy.
- If the retrieved rules are insufficient, return `needs_review`.
- Do not invent company policy or rely on outside legal knowledge.
- Every `non_compliant` finding must cite at least one matched rule ID.
- Copy relevant clause and rule excerpts verbatim where fields require evidence.
- Use `temperature=0` for repeatability.

## Extraction Schemas

### `contract_clauses`
 
| Field | Type | Description |
|-------|------|-------------|
| `clause_type` | `str \| null` | Clause category, such as `termination`, `liability`, `privacy`, or `payment` |
| `text` | `str` | Verbatim clause text |

### `contract_metadata`
Structured information extracted by the LLM from a contract document.
``doc_id`` is injected by the LLMExtractor; do not include it here.

| Field | Type | Description |
|-------|------|-------------|
| `contract_type` | `str \| null` | Contract category |
| `parties` | `list[Party]` | Named parties and roles |
| `effective_date` | `str \| null` | Start date in `YYYY-MM-DD` format |
| `expiry_date` | `str \| null` | End date in `YYYY-MM-DD` format |
| `contract_value` | `float \| null` | Total monetary value when stated |
| `currency` | `str \| null` | ISO 4217 currency code |
| `governing_law` | `str \| null` | Governing law clause or jurisdiction |
| `termination_notice_days` | `int \| null` | Notice period for termination |

### `clause_compliance_findings`

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Source contract document ID |
| `clause_id` | `str` | Reviewed clause ID |
| `clause_type` | `str \| null` | Reviewed clause category |
| `status` | `str` | `compliant`, `non_compliant`, `needs_review`, or `not_applicable` |
| `severity` | `str` | `low`, `medium`, `high`, or `critical` |
| `summary` | `str` | Short human-readable finding |
| `contract_clause_text` | `str` | Verbatim reviewed clause text |
| `matched_rule_quotes` | `list[str]` | Verbatim excerpts from matched rules |
| `reasoning` | `str` | Explanation grounded in matched rules |
| `recommended_redline` | `str \| null` | Suggested replacement or fallback language |
| `confidence` | `float` | Judge confidence from 0.0 to 1.0 |

## Project structure

```text
contract_compliance_demo/
├── README.md
├── demo.py                    # interactive demo script
├── rules_data.py              # sample company rule documents
├── contracts_data.py          # sample incoming contracts
├── schema.py                  # Pydantic models for clauses, metadata, findings
└── compliance_skill.py        # deterministic compliance-check skill wrapper
```

## Design notes

**Why one application?** Rules and contracts are part of the same compliance
workspace. Keeping them in one app avoids cross-app collection access and lets
the query runner see all relevant structured and vector collections.
`metadata.doc_type` controls which pipeline steps apply to each document type.

**Why metadata-based routing?** Rules documents and contracts have different
ingestion behavior but share storage, retrieval, and reporting. Conditional
pipeline steps keep routing explicit without introducing a full workflow engine.

**Why a skill and not a normal ingestion step?** Compliance checking reads
previously extracted clause records, searches rule chunks, calls an LLM judge,
and writes findings. That is a cross-collection workflow over persisted records,
not a simple document ingestion step.

**Clause extraction** uses a dedicated extractor that produces one record per
clause rather than one record per document. The extractor labels each clause by
type and copies text verbatim so stored text can serve as evidence.

## Known limitations

- Clause extraction quality depends on contract formatting and OCR quality.
- Semantic search may retrieve incomplete policy context for broad or ambiguous clauses.
- The LLM judge is policy-assistive and should not be treated as legal advice.
- Rule versioning is simplified to a `ruleset_id` in the demo.
- The metadata-based `when` syntax shown here describes the intended design; it
  requires pipeline step filtering support in CogBase before it can be purely
  config-driven.
