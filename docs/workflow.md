# Workflows

Workflows are named, on-demand analytical computations that run over already-ingested collections. They sit between the ingestion pipeline (document-time, automatic) and skills (query-time, LLM-callable):

| Concept | Trigger | Input | Output |
|---|---|---|---|
| `IngestionPipeline` | document arrives | raw `Document` | vector + structured collections |
| **`Workflow`** | API call or after ingest | typed params | derived records + SSE stream |
| `Skill` | LLM during query | arbitrary dict | arbitrary dict |

The contract compliance demo is the canonical example: after clauses are extracted from a contract, a workflow retrieves matching policy rules, calls an LLM judge for each clause, and saves findings to a separate collection.

---

## Concepts

### Steps

A workflow is a sequential list of steps. Each step is either a **tool call** (leaf) or a **foreach loop** (container):

**Tool step** — calls one built-in tool and stores its output under `steps.<id>`:

```yaml
- id: load_clauses
  tool: structured-query
  collection: contract_clauses
  filters:
    doc_id: "{{ input.doc_id }}"
```

**Foreach step** — iterates over a list, running nested steps for each item. Each iteration gets its own `steps` namespace; outputs from one iteration don't bleed into the next:

```yaml
- id: review_each_clause
  foreach: "{{ steps.load_clauses.records }}"
  steps:
    - id: retrieve_rules
      ...
    - id: judge
      ...
```

### Template expressions

Step parameters are Jinja2 templates rendered with [Jinja2's `NativeEnvironment`](https://jinja.palletsprojects.com/en/stable/nativetypes/), so a `{{ expr }}` that resolves to a list returns an actual Python list, not a string. Three namespaces are available:

| Variable | Type | Description |
|---|---|---|
| `input` | `dict` | Parameters passed when the workflow was invoked |
| `steps.<id>` | `dict` | Output of a previously completed step |
| `item` | any | Current element inside a `foreach` loop |

Attribute access follows Jinja2 dot notation: `steps.load_clauses.records`, `item.clause_type`.

### Step outputs

Each tool writes its result into `steps.<id>` so later steps can reference it:

| Tool | Output keys |
|---|---|
| `structured-query` | `records` — `list[dict]` |
| `vector-search` | `chunks` — `list[Chunk]` |
| `llm-structured` | `output` — Pydantic model instance |
| `structured-save` | `records` — list of saved records (also streamed to caller) |

Only `structured-save` streams: one dict is yielded per saved record, in order.

### Triggers

```yaml
trigger:
  type: manual          # default — only runs when explicitly called via API
```

```yaml
trigger:
  type: after_ingest
  when:
    metadata:
      doc_type: contract  # only fires for documents with this metadata
```

`after_ingest` workflows run as a background task after each successful ingest. They must derive their input params from structured records produced by ingestion. Failures are logged and do not affect the ingest response.

```yaml
trigger:
  type: after_ingest
  params_from_collection:
    collection: facts
    filters:
      doc_id: "{{ doc.doc_id }}"
    params:
      issue: "{{ record.issue }}"
```

This queries `facts` after the document is ingested and starts one workflow per distinct rendered param set.

### Output collections

Structured collections written to by `structured-save` steps are declared at the **app level** under `structured_collections` in `config.yaml`, not inside individual workflow blocks. The factory creates all collections at startup (idempotent):

```yaml
structured_collections:
  - name: clause_compliance_findings
    schema: clause_compliance_findings_schema.json   # JSON Schema file in ZIP bundle
    primary_fields: [finding_id]
    description: "Clause-level compliance findings"

workflows:
  - name: check-contract-compliance
    steps:
      - id: save_finding
        tool: structured-save
        collection: clause_compliance_findings   # references the collection above
        ...
```

---

## Built-in tools

### `structured-query`

Queries a structured collection with equality filters. All filters are ANDed.

```yaml
- id: load_clauses
  tool: structured-query
  collection: contract_clauses
  filters:
    doc_id: "{{ input.doc_id }}"
    clause_type: liability        # literal value, no template needed
```

Output: `steps.<id>.records` — `list[dict]`

### `vector-search`

Embeds `query` and searches a vector collection.

```yaml
- id: retrieve_rules
  tool: vector-search
  collection: rule_chunks
  query: "{{ item.clause_type }}\n{{ item.text }}"
  top_k: 5
```

Output: `steps.<id>.chunks` — `list[Chunk]`

### `llm-structured`

Calls the LLM with a system prompt and a JSON-serialised `input` dict, then validates the response against `output_schema`.

```yaml
- id: judge
  tool: llm-structured
  prompt: compliance_judge_prompt.txt      # file in ZIP bundle; resolved at upload time
  input:
    clause: "{{ item }}"
    rules: "{{ steps.retrieve_rules.chunks }}"
  output_schema: clause_compliance_findings_schema.json
```

- **`prompt`** becomes the system message. It may contain `{{ variable }}` references if needed.
- **`input`** values are rendered as native Python types and JSON-serialised into the user message, followed by the schema hint.
- **`output_schema`** is a JSON Schema (resolved from a file reference). The response is validated against it and returned as a Pydantic model instance at `steps.<id>.output`.

LLM is called with `temperature=0.0`.

### `structured-save`

Upserts records into a structured collection and streams each one to the caller.

```yaml
- id: save_finding
  tool: structured-save
  collection: clause_compliance_findings
  records: ["{{ steps.judge.output }}"]
```

`records` is a list of template expressions. Each expression should resolve to a Pydantic model instance (typically the `output` of an `llm-structured` step). Idempotency is handled by the collection's `primary_fields` — re-running the workflow overwrites prior findings with the same primary key.

Output: `steps.<id>.records` — list of saved records (also yielded by the runner).

---

## Full example: contract compliance check

```yaml
workflows:
  - name: check-contract-compliance
    trigger:
      type: manual
    input_schema:
      doc_id: string
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
            records: ["{{ steps.judge.output }}"]
```

---

## API endpoints

```
GET  /applications/{name}/workflows
     → list registered workflow names

POST /applications/{name}/workflows/{workflow_name}/run
     Body: {"params": {"doc_id": "contract-001"}}
     → {"workflow": "...", "records": [...], "total": N}

POST /applications/{name}/workflows/{workflow_name}/stream
     Body: {"params": {"doc_id": "contract-001"}}
     → SSE stream
       data: {"record": {...}}
       data: {"record": {...}}
       data: [DONE]
```

---

## Module layout

```
cogbase/workflows/
├── __init__.py          # exports WorkflowRunner
├── runner.py            # WorkflowRunner — sequential step executor
├── context.py           # Jinja2 NativeEnvironment template rendering
└── tools/
    ├── __init__.py      # run_tool() dispatcher
    ├── structured_query.py
    ├── vector_search.py
    ├── llm_structured.py
    └── structured_save.py
```

Config models live in `cogbase/config/config.py`: `WorkflowConfig`, `WorkflowTriggerConfig`, `WorkflowStepBase`, and the four typed leaf step configs (`StructuredQueryStepConfig`, `VectorSearchStepConfig`, `LLMStructuredStepConfig`, `StructuredSaveStepConfig`) plus `ForeachStepConfig`. `WorkflowLeafStepConfig` is a discriminated union over the four leaf types. `WorkflowStepConfig = WorkflowLeafStepConfig | ForeachStepConfig`. The factory in `api/factory.py` creates all structured collections (including workflow outputs) and builds `WorkflowRunner` instances. `CogBaseApp` holds the runners and fires `after_ingest` triggers.
