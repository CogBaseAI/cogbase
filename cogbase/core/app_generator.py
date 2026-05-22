"""App generator — agentic, conversational config.yaml creation.

The LLM drives the conversation via schema/config tools. Schemas and configs
validate server-side and return errors for the LLM to fix. The client owns the
full message history (role: user/assistant only); tool call/result messages live
only within a single server turn.
"""

from __future__ import annotations

import copy
import json
import logging

import yaml

from cogbase.config.config import AppConfig, StructuredCollectionConfig, WorkflowConfig
from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema
from cogbase.llms.base import LLMBase, ToolDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_PROPOSE_APP_CONFIG_TOOL: ToolDefinition = {
    "name": "propose_app_config",
    "description": (
        "Generate and validate the complete app config from the confirmed field list and "
        "workflow design. Call this once the user has confirmed all fields (and any workflow "
        "steps). Returns a brief summary on success, or an error message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "needs_workflow": {
                "type": "boolean",
                "description": (
                    "Set to true if the confirmed design includes a workflow "
                    "(analytical fan-out over a collection). False for apps that only "
                    "need ingestion and query."
                ),
            },
        },
        "required": ["needs_workflow"],
        "additionalProperties": False,
    },
}

GENERATOR_TOOLS: list[ToolDefinition] = [
    _PROPOSE_APP_CONFIG_TOOL,
]

# generate the extraction schema that llm will use to extract data from a document,
# e.g. ExtractorConfig.extraction_schema, no other data in AppConfig.
_EXTRACTION_SCHEMA_AGENT_SYSTEM_PROMPT = """\
You are a CogBase extraction schema designer. Given a conversation about building \
a CogBase application, produce JSON Schema definitions only for structured \
collections produced by pipeline extract-structured steps. Generate schemas that \
match exactly the ingestion fields the user has already confirmed in the \
conversation — do not add, remove, or rename fields.

CogBase has three store types — design schemas only for structured collections:
- Structured collections: discrete extractable facts for filtered/exact lookup (what you design here)
- Vector/chunk collections: full-text passages for semantic search (handled automatically by chunk-embed-upsert)
- Document collections: LLM summaries for high-level queries (handled automatically by document-embed-upsert)
Do NOT include fields like document_text, full_text, body, or summary — those are covered \
by the other two collections automatically.

Do NOT design workflow output collections here. Collections written by \
structured-save or used as llm-structured output_schema are handled by a separate \
workflow schema step.

Use domain knowledge to propose sensible fields — do not wait for the user to enumerate \
them. For example, if the application is a contract analyst, include fields like \
vendor_name, effective_date, expiry_date, total_value, and governing_law without being asked.

Output ONLY a YAML mapping of collection_name → JSON Schema object. No explanation, \
no markdown fences, just the raw YAML.

Schema rules:
- Top-level keys are collection names (snake_case)
- Each collection must be type: object with a non-empty properties block
- Do NOT include doc_id — it is injected automatically
- Optional/nullable scalars: anyOf: [{type: <T>}, {type: "null"}]
- List fields: type: array, items: {...}, default: []
- Nested objects: type: object with inline properties
- Add a description to every field

Example output:
  contracts:
    type: object
    properties:
      vendor_name:
        anyOf: [{type: string}, {type: "null"}]
        description: Name of the vendor
      effective_date:
        anyOf: [{type: string}, {type: "null"}]
        description: Contract start date (ISO 8601)
      line_items:
        type: array
        items:
          type: object
          properties:
            description: {type: string, description: Item description}
            amount: {anyOf: [{type: number}, {type: "null"}], description: Amount in USD}
        description: Contract line items
        default: []\
"""

_WORKFLOW_SCHEMA_AGENT_SYSTEM_PROMPT = """\
You are a CogBase workflow schema designer. Given a conversation about building \
a CogBase application and the validated pipeline record schemas below, produce \
JSON Schema definitions only for workflow output collections written by \
structured-save and used as llm-structured output_schema.

Generate schemas that match exactly the workflow output fields the user has \
already confirmed in the conversation — do not add, remove, or rename fields.

Workflow output schemas describe records created by workflow logic. They may \
include provenance fields (doc_id, source record identifiers), status fields, \
evidence references, and LLM judgment fields.

Use the validated pipeline record schemas to understand what fields and identifiers \
are available in each upstream collection. If a workflow iterates over records from \
a pipeline collection, include the fields needed to trace each output record back to \
its source — these may include doc_id, per-record identifiers (e.g. clause_id, \
rule_id), and any other provenance fields present in the source schema.

Output ONLY a YAML mapping of workflow_output_collection_name → JSON Schema object. \
If the confirmed design has no workflow output collections, output an empty YAML \
mapping: {}

Schema rules:
- Top-level keys are workflow output collection names (snake_case)
- Each non-empty collection must be type: object with a non-empty properties block
- Include a stable identifier field when the workflow creates independently saved records \
  (for example finding_id, review_id, or the source record's own identifier like clause_id)
- Include doc_id when it is useful provenance
- Optional/nullable scalars: anyOf: [{type: <T>}, {type: "null"}]
- List fields: type: array, items: {...}, default: []
- Nested objects: type: object with inline properties
- Add a description to every field

Example output:
  clause_compliance_findings:
    type: object
    properties:
      clause_id:
        type: string
        description: Identifier of the reviewed clause from contract_clauses
      doc_id:
        type: string
        description: Source contract document identifier
      status:
        type: string
        description: Compliance status
      severity:
        anyOf: [{type: string}, {type: "null"}]
        description: Severity when non-compliant
      summary:
        type: string
        description: Short compliance finding summary\
"""

_MAX_SCHEMA_RETRIES = 3
_MAX_CONFIG_RETRIES = 3

_PIPELINE_CONFIG_AGENT_SYSTEM_PROMPT = f"""\
You are a CogBase pipeline config generator. Given a conversation about building a CogBase \
application and the validated pipeline extraction schemas injected below, produce the data-model \
section of a valid config.yaml: name, vector_collections, structured_collections \
(pipeline-backed collections only), and pipelines.

Output ONLY the raw YAML — no explanation, no markdown fences.

Use the extraction_schema values from the "Validated pipeline extraction schemas" section verbatim — \
do not rewrite or reformat them.

Do NOT include workflow output collections (structured-save targets) or workflows — those are \
generated in a separate step.

## Rules
1. name must be kebab-case (lowercase, alphanumeric, hyphens only)
2. chunk-embed-upsert is always the first pipeline step
3. Do NOT include doc_id in any extraction_schema — it is injected automatically by the \
   pipeline. For record_mode: many, also do NOT include the id_field (e.g. clause_id, rule_id, \
   item_id) in the extraction_schema — it is injected automatically too. Including either field \
   will fail server-side schema validation.
4. For every extract-structured step, choose record_mode based on how many records the LLM \
   should return per document:
   - record_mode: one (default) — the LLM returns ONE record for the whole document. \
     Use when the document yields exactly one entity of this type. Examples: contract-level \
     metadata (one set of header-level facts per contract), board update KPIs \
     (one snapshot per deck), deal memo summary (one memo per document).
   - record_mode: many — the LLM returns MULTIPLE records as a list. Use when the document \
     contains a list of distinct entities of this type. Examples: clauses in a contract, \
     rules/prohibitions/requirements in a policy document, line items in an invoice, \
     findings in an audit report, employees in a roster, transactions in a statement. \
     When record_mode is many, ALSO set: response_field (the array key in the extractor \
     output, e.g. clauses, rules, items), id_field (the per-record identifier, e.g. \
     clause_id, rule_id, item_id), and id_template (e.g. "{{doc_id}}__{{index:04d}}"). \
     The id_field is injected by the pipeline using id_template — it must NOT appear in \
     extraction_schema.properties.
   Primary signal — multiplicity annotation in the conversation: when the field-proposal \
   turn labels a collection as "many records per document" or "one per <entity>" \
   (e.g. "one per clause", "one per rule", "one per line item"), ALWAYS set \
   record_mode: many for that collection. Never override this with record_mode: one. \
   Fallback heuristic when no annotation is present: if a natural plural \
   ("the clauses", "the rules", "the line items") describes what the schema captures, \
   choose record_mode: many. If the schema is a flat set of header-level facts about \
   the document itself, choose record_mode: one. Getting this wrong collapses many \
   entities into a single record (loses data) or wraps a singleton in a list for no reason.
5. All content is INLINE — do not use .json or .txt filenames as values anywhere
6. Pipeline step collections must exactly match declared vector/structured collection names
7. Use snake_case for all collection names and field names
8. Every pipeline must have a routing_description — a plain-language sentence describing \
   which documents belong in that pipeline (used by LLM routing to classify documents)
9. YAML quoting: any plain string value that contains ": " (colon followed by a space) \
   will break YAML parsing. Always wrap such values in double quotes. \
   Example — BAD:  description: Facts extracted from contracts: vendor, dates, value. \
   Example — GOOD: description: "Facts extracted from contracts: vendor, dates, value."

## Config format

{AppConfig.config_format_prompt()}

## Example — two document types, shared vector collections

name: vc-portfolio

vector_collections:
  - name: portfolio_chunks
    description: "Full-text passages from all portfolio documents."
  - name: portfolio_summaries
    description: "One-per-document summaries of portfolio documents."

structured_collections:
  - name: portfolio_kpis
    description: "Financial and operational KPIs extracted from board decks and LP updates. Use for exact lookups and cross-company comparisons."

pipelines:
  # board-update pipeline: chunk + extract KPIs + summarize
  - name: board-update
    routing_description: "Board decks and LP updates from portfolio companies containing financial KPIs, headcount, milestones, and risk sections."
    match:
      metadata:
        doc_type: board_update
    steps:
      - tool: chunk-embed-upsert
        collection: portfolio_chunks

      - tool: extract-structured
        collection: portfolio_kpis
        extractor:
          type: llm
          extraction_schema: '{{"type":"object","properties":{{"company_name":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Portfolio company name"}},...}}}}'
          prompt: |
            You are a VC portfolio analyst. Extract structured financial and
            operational KPIs from the board deck or LP update provided.

            Rules:
            - Use null for any field not present in the document. Never invent or estimate values.
            - All monetary values must be plain numbers with no currency symbols or abbreviations (e.g. 2400000, not "$2.4M").
            - All percentages must be plain numbers (e.g. 85.0, not "85%").
            - reporting_period must be in "Q# YYYY" format (e.g. "Q3 2024"). If only a year is given, use "FY YYYY".
            - key_milestones and notable_risks must be concise verbatim phrases from the document, not paraphrases.
            - Return ONLY the JSON object — no explanation, no markdown fences.

      - tool: document-embed-upsert
        collection: portfolio_summaries
        doc_prompt: >
          Summarize this board deck or LP update. Include: company name,
          reporting period, key financials, headcount, milestones, and risks.

  # deal-memo pipeline: chunk + summarize only — no KPI extraction from pitch decks
  - name: deal-memo
    routing_description: "Investment memos and pitch decks for prospective deals containing investment thesis, deal terms, and sector analysis."
    match:
      metadata:
        doc_type: deal_memo
    steps:
      - tool: chunk-embed-upsert
        collection: portfolio_chunks

      - tool: document-embed-upsert
        collection: portfolio_summaries
        doc_prompt: >
          Summarize this investment memo or pitch deck. Include: company name,
          stage, sector, investment thesis, key risks, and deal terms if present.

## Example — pipeline section of a workflow app (workflow output collection excluded)

# For apps with workflows, generate only the pipeline-backed structured_collections.
# The workflow output collection (clause_compliance_findings) is added in the next step.

name: contract-compliance

vector_collections:
  - name: rule_chunks
    description: "Company policy passages and standards used as evidence for compliance judgments."
  - name: contract_chunks
    description: "Contract text passages for detailed questions about specific contract terms or clauses."

structured_collections:
  - name: contract_metadata
    description: "Key facts per contract: parties, dates, value, governing law, termination notice period. ONE record per contract document."
  - name: contract_clauses
    description: "Individual clauses extracted from contracts. MANY records per contract — each record is one clause with its type and verbatim text."

pipelines:
  - name: rules
    routing_description: "Company policy documents, internal standards, compliance guidelines, and fallback positions that define rules contracts must be checked against."
    match:
      metadata:
        doc_type: rules
    steps:
      - tool: chunk-embed-upsert
        collection: rule_chunks

  - name: contracts
    routing_description: "Vendor contracts, commercial agreements, and service agreements to be reviewed for compliance against company policy."
    match:
      metadata:
        doc_type: contract
    steps:
      - tool: chunk-embed-upsert
        collection: contract_chunks

      - tool: extract-structured
        collection: contract_metadata
        extractor:
          type: llm
          extraction_schema: '{{"type":"object","properties":{{"contract_type":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Contract category"}},...}}}}'
          prompt: |
            You are a legal contract analyst. Extract key contract-level facts.
            Return ONLY the JSON object — no explanation, no markdown fences.

      - tool: extract-structured
        collection: contract_clauses
        extractor:
          type: llm
          extraction_schema: '{{"type":"object","properties":{{"clause_type":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Clause category"}},"text":{{"type":"string","description":"Verbatim clause text"}}}}}}'
          record_mode: many
          response_field: clauses
          id_field: clause_id
          id_template: "{{doc_id}}__{{index:04d}}"
          prompt: |
            You are a legal contract analyst. Extract every distinct clause.
            Return ONLY the JSON object — no explanation, no markdown fences.\
"""

_WORKFLOW_CONFIG_AGENT_SYSTEM_PROMPT = (
    """\
You are a CogBase workflow config generator. Given a validated pipeline config, \
full pipeline record schemas, and validated workflow output schemas — all injected \
below — produce the workflow additions to the config.

Output ONLY the raw YAML for two sections: structured_collections (workflow output \
collections only) and workflows. Do NOT output name, vector_collections, pipelines, \
or pipeline-backed structured_collections — those are already validated and locked.

Use the output_schema values from "Validated workflow output schemas" verbatim in \
llm-structured steps.

The "Validated pipeline record schemas" section shows the full stored record schema \
for each pipeline-backed structured collection — extraction fields plus injected \
doc_id and, for RecordMode.MANY collections, the per-record id_field (e.g. clause_id). \
Use these to correctly set primary_fields and to pass the right identifiers through \
the workflow steps.

## Rules
1. output_schema in llm-structured workflow steps must be an inline JSON string — use \
   the exact value from "Validated workflow output schemas".
2. All content is INLINE — do not use .json or .txt filenames as values anywhere.
3. Do NOT set schema on workflow output structured_collections — schema is injected \
   automatically from the validated workflow output schemas. Output only name and \
   description for each workflow output collection.
4. structured-save depends on the upstream llm-structured step that produces its records. \
   Every field listed in a structured-save `primary_fields` MUST also be declared as a \
   property in that upstream llm-structured `output_schema`. structured-save persists \
   exactly what the LLM produced — a primary field absent from `output_schema` will be \
   missing on every saved record and the workflow will fail validation. Concretely, when \
   `primary_fields` is `[doc_id, clause_id]` (or similar provenance identifiers like \
   `finding_id`, `company_id`): \
   (a) the upstream llm-structured `output_schema.properties` MUST include each of those \
       fields, \
   (b) the llm-structured `input` block MUST expose the source values (e.g. pass the whole \
       `item` so `item.doc_id` / `item.clause_id` are visible to the LLM), and \
   (c) the llm-structured `prompt` MUST instruct the LLM to copy those identifier fields \
       verbatim from the input into its output. \
   The same rule applies to the workflow output collection's primary_fields — they are \
   derived from these structured-save primary_fields at validation time, so the workflow \
   output schema must also include them.
5. Every workflow must have a params_from_collection block that derives input params from a \
   structured collection. Use filters to select by doc_id and params to expose the values \
   the workflow steps reference via {{ input.* }}.
6. Avoid bulk context in judgment workflows:
   - Never use structured-query with empty filters to load an entire collection and
     pass all records to an llm-structured step. This floods the model context and
     can cause empty or low-quality output.
   - When judging a record against unstructured reference material, such as policy,
     rules, guidelines, or source-document text, retrieve targeted context inside
     the foreach loop using vector-search and pass only top-k chunks.
   - When judging relationships among structured records, such as contradictions,
     duplicate facts, evidence gaps, reconciliations, or cross-record consistency,
     use structured-query with selective filters such as issue, entity, date range,
     doc_id, account_id, or contract_id to load only the relevant peer records.
   - Never filter a collection from one pipeline by a doc_id derived from a different
     pipeline. For example, if a workflow iterates over vendor contract clauses
     (where input.doc_id is a contract ID), do NOT query policy_rules or policy_documents
     with filters: {doc_id: '{{ input.doc_id }}'} or {doc_id: '{{ item.doc_id }}'} —
     policy records carry policy doc_ids, not contract doc_ids, so this filter always
     returns zero results. Use vector-search instead to retrieve cross-pipeline reference
     material by semantic similarity (e.g. query policy_chunks by clause text or topic).
7. YAML quoting: any plain string value that contains ": " (colon followed by a space) \
   will break YAML parsing. Always wrap such values in double quotes. \
   Example — BAD:  description: Findings produced by the workflow: compliance status, severity. \
   Example — GOOD: description: "Findings produced by the workflow: compliance status, severity."

## Config format — workflow output collection

"""
    + StructuredCollectionConfig.config_format_prompt()
    + """

## Config format — workflow

"""
    + WorkflowConfig.config_format_prompt()
    + """

## Example output

# Only these two sections — nothing else
structured_collections:
  - name: clause_compliance_findings
    description: "Clause-level compliance findings produced by the compliance workflow. Each record captures whether a clause complies with company policy, with severity and reasoning."

workflows:
  - name: check-contract-compliance
    trigger:
      type: manual
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
            query: "{{ item.clause_type }}\\n{{ item.text }}"
            top_k: 5

          - id: judge
            tool: llm-structured
            prompt: |
              You are a contract compliance reviewer. Judge whether the clause complies
              with company policy using ONLY the policy excerpts provided.
              Rules:
              - Ground every finding exclusively in the provided excerpts.
              - If excerpts are insufficient, set status=needs_review.
              - Copy clause_id and doc_id verbatim from the input clause into your output.
              - Return ONLY valid JSON — no markdown fences, no explanation.
            input:
              clause: "{{ item }}"
              rules: "{{ steps.retrieve_rules.chunks }}"
            output_schema: '<verbatim JSON from Validated workflow output schemas — clause_compliance_findings>'

          - id: save_finding
            tool: structured-save
            collection: clause_compliance_findings
            primary_fields: [doc_id, clause_id]
            records:
              - "{{ steps.judge.output }}"\
"""
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an agentic CogBase application generator. Help the user build a complete, \
correct CogBase app configuration through natural conversation. You drive the process.

CogBase applications ingest documents, extract structured facts with an LLM, and answer \
natural-language questions via semantic search and structured lookup.

## Core concepts

**Pipeline steps and stores** — three step types, each writing to a different store:
- `chunk-embed-upsert` → vector collection: overlapping text passages for semantic search
- `extract-structured` → structured collection: discrete typed facts for filtered/exact lookup
- `document-embed-upsert` → document (vector) collection: one LLM summary per document for high-level queries
Full text and summaries are covered automatically — structured collections hold only discrete extracted facts.

**Multiple pipelines** — an app can declare multiple named pipelines when different document \
types need different processing (different extraction fields, different step sets). Pipelines \
can share vector collections. CogBase routes each document to the right pipeline automatically \
using one of three strategies — you do not need to ask about metadata availability to decide \
whether multiple pipelines make sense:
- Metadata routing (`routing_strategy: metadata`): each pipeline declares a `match` block \
  (e.g. `doc_type: board_update`); the first matching pipeline wins. Use this when the user \
  controls document upload and can reliably supply metadata.
- LLM routing (`routing_strategy: llm`): CogBase reads each pipeline's `routing_description` \
  and asks an LLM to classify the document. No metadata needed — works on raw document content.
- Auto routing (`routing_strategy: auto`, the default): tries metadata matching first; if no \
  pipeline matches, falls back to LLM routing. Best default when metadata may or may not be present.

Ask whether different document types need different pipeline treatment (different steps or \
extracted fields). If yes, configure multiple pipelines. Use `routing_strategy: auto` unless \
the user says they will always supply reliable metadata (then use `metadata`) or explicitly \
wants LLM-only routing (then use `llm`).

**Workflows** — YAML-declared analytical pipelines that fan out over all records in a collection \
(e.g. "flag every contract expiring before Q2", "rank all portfolio companies by ARR"). \
Use workflows when an example question requires scanning the whole collection, not just retrieving \
a single document. Single-document retrieval and lookup queries are handled by the query runner \
without a workflow.

## How to work

1. Ask targeted questions — no more than 2-3 per turn — to understand:
   - What the documents are about (domain and subject matter)
   - Whether there are multiple document types that need different pipeline treatment
   - What kinds of queries users will run:
       * Exact/filtered lookup over extracted facts → extract-structured + structured collection
       * Semantic search over document text → chunk-embed-upsert + vector collection
       * High-level summary or topic queries → document-embed-upsert + document collection
       * Analytical fan-out over all records (e.g. "flag all X that…") → workflow

2. Once you understand the domain, propose the target fields for each structured collection
   as a bullet list. Use nested bullets for object or array fields.

   For each collection, start the heading with a multiplicity annotation that tells the
   config generator how many records the LLM should produce per document:
   - *(one record per document)* — a single set of header-level facts about the document
     (e.g. contract-level metadata, board update KPIs, deal memo summary).
   - *(many records per document — one per <entity>)* — a list of distinct entities
     within the document (e.g. clauses in a contract, rules in a policy document, line
     items in an invoice). Use this whenever the user's goal involves acting on individual
     instances. Any task that says "check each clause", "flag every rule", or "review each
     line item" always implies many records per document — never collapse these into one.

   Example:

   **contract_metadata** *(one record per contract)*
   - vendor_name — name of the vendor
   - effective_date — contract start date (ISO 8601)
   - payment_terms
     - schedule - payment schedule, e.g. "net-30", "monthly", "upfront", "milestone-based"
     - late_penalty - penalty or interest rate for late payment, verbatim if present
   - key_terms (list) - significant defined terms, unusual provisions

   **contract_clauses** *(many records per contract — one per clause)*
   - clause_type — category: liability, indemnification, termination, payment, etc.
   - text — verbatim clause text

   Use domain knowledge to propose sensible fields — do not ask the user to enumerate them.
   Ask the user to confirm, add, remove, or rename fields. Revise conversationally until confirmed.

   If any queries require analytical fan-out (e.g. "check every clause for compliance",
   "rank all companies by ARR"), also design the workflow before calling propose_app_config:
   - Sketch the step sequence: what records to load, what to iterate over, what LLM judgment
     to apply, and where to save results.
   - For judgment workflows, choose retrieval based on the comparison target:
     * If judging each record against unstructured reference material such as policy,
       rule, or source-document text, retrieve targeted context inside the foreach loop
       using vector-search.
     * If judging relationships among structured records, such as contradictions
       between facts, use structured-query with selective filters such as issue,
       entity, date range, or doc_id to load only the relevant peer records.
     * Never load an entire collection via structured-query with empty filters and
       dump all records into one LLM step.
   - Identify the workflow output collection (e.g. "compliance_findings") and add its fields
     to the proposed field list — its schema will be generated after the pipeline extraction schemas.
   - Confirm the workflow design with the user.

   Once the field list (and any workflow design) is confirmed, call propose_app_config.
   On success, present the result to the user with a plain-language explanation of what was
   configured and why. On failure, explain what went wrong and ask for any clarification needed."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record_schema(extraction_schema: dict, id_field: str | None = None) -> dict:
    """Add required doc_id (and optional id_field) to produce the record schema.

    Mirrors the field injection done by ``LLMExtractor`` at extraction time:
    RecordMode.ONE collections get ``doc_id``; RecordMode.MANY collections get
    ``doc_id`` + ``id_field``.
    """
    record = copy.deepcopy(extraction_schema)
    props = record.setdefault("properties", {})
    required = record.setdefault("required", [])
    if id_field:
        props[id_field] = {"type": "string", "description": "record identifier"}
        if id_field not in required:
            required.insert(0, id_field)
    props["doc_id"] = {"type": "string", "description": "document identifier"}
    if "doc_id" not in required:
        required.insert(0, "doc_id")
    return record


def _collect_save_targets(steps: list, targets: set[str]) -> None:
    """Recursively collect structured-save collection names from workflow steps."""
    for step in steps:
        if step.get("tool") == "structured-save":
            coll = step.get("collection")
            if coll:
                targets.add(coll)
        inner = step.get("steps")
        if inner:
            _collect_save_targets(inner, targets)


def _inject_pipeline_record_schemas(config_dict: dict) -> None:
    """Set pipeline structured collection schema and primary_fields from extractor config.

    Pipeline extract-structured targets store records, not raw extraction objects,
    so their collection schema is extraction_schema + injected doc_id (+ id_field
    for record_mode=many). primary_fields is derived to match: ``[doc_id]`` for
    RecordMode.ONE, ``[doc_id, id_field]`` for RecordMode.MANY.
    """
    ext_info: dict[str, tuple[dict, str | None]] = {}
    for pipeline in config_dict.get("pipelines", []):
        for step in pipeline.get("steps", []):
            if step.get("tool") != "extract-structured":
                continue
            collection = step.get("collection")
            extractor = step.get("extractor") or {}
            ext_schema_str = extractor.get("extraction_schema", "")
            if not (collection and ext_schema_str):
                continue

            try:
                ext_schema = json.loads(ext_schema_str)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.exception(
                    "pipeline collection=%s, extraction_schema is not valid JSON: %s",
                    collection,
                    ext_schema_str,
                )
                raise ValueError(
                    f"pipeline collection '{collection}': extraction_schema is not valid JSON: {exc}"
                ) from exc

            record_id_field = (
                extractor.get("id_field") if extractor.get("record_mode") == "many" else None
            )
            if record_id_field and record_id_field in ext_schema.get("properties", {}):
                # LLM mistakenly included the id_field in the extraction schema even though
                # it is injected automatically via id_template. Strip it here so the extractor
                # doesn't ask the LLM to produce it, and write the cleaned schema back so the
                # stored config is consistent.
                ext_schema["properties"].pop(record_id_field)
                if record_id_field in ext_schema.get("required", []):
                    ext_schema["required"].remove(record_id_field)
                extractor["extraction_schema"] = json.dumps(ext_schema, separators=(",", ":"))

            ext_info[collection] = (ext_schema, record_id_field)

    for sc in config_dict.get("structured_collections", []):
        name = sc.get("name")
        if name not in ext_info:
            continue
        ext_schema, record_id_field = ext_info[name]
        sc["schema"] = json.dumps(
            _make_record_schema(ext_schema, id_field=record_id_field),
            separators=(",", ":"),
        )
        sc["primary_fields"] = ["doc_id"] + ([record_id_field] if record_id_field else [])


def _inject_workflow_output_schemas(
    config_dict: dict,
    workflow_schemas: dict[str, str] | None = None,
) -> None:
    """Set workflow structured-save target schemas from validated workflow schemas.

    Always overwrite any inline `schema` on a save-target collection: the config
    LLM is instructed not to author one, and `workflow_schemas` has been validated
    by `_validate_workflow_output_schema` whereas an inline schema has not.
    """
    if not workflow_schemas:
        return

    save_targets: set[str] = set()
    for workflow in config_dict.get("workflows", []):
        _collect_save_targets(workflow.get("steps", []), save_targets)
    for sc in config_dict.get("structured_collections", []):
        name = sc.get("name")
        if name in save_targets and name in workflow_schemas:
            try:
                schema_dict = json.loads(workflow_schemas[name])
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"workflow schema for collection '{name}' is not valid JSON: {exc}"
                ) from exc
            sc["schema"] = json.dumps(schema_dict, separators=(",", ":"))


def _validate_extraction_schema(schema_dict: dict, collection_name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema_dict, dict):
        return [f"[{collection_name}] must be a JSON Schema object (mapping)"]
    props = schema_dict.get("properties", {})
    if "doc_id" in props:
        errors.append(
            f"[{collection_name}] doc_id must not appear in the extraction schema"
            " — it is injected automatically"
        )
    if not props:
        errors.append(f"[{collection_name}] schema must have at least one field in 'properties'")
    if errors:
        return errors
    try:
        build_model_from_json_schema(schema_dict, model_name=collection_name)
    except Exception as exc:
        errors.append(f"[{collection_name}] invalid JSON Schema: {exc}")
    return errors


def _validate_workflow_output_schema(schema_dict: dict, collection_name: str) -> list[str]:
    # Stable-identifier presence is not enforced here: the validator runs before
    # config generation, so we cannot yet tell which collections are
    # structured-save targets. Identifier guidance lives in the workflow schema
    # prompt and is reinforced by AppConfig.primary_fields validation downstream.
    errors: list[str] = []
    if not isinstance(schema_dict, dict):
        return [f"[{collection_name}] must be a JSON Schema object (mapping)"]
    props = schema_dict.get("properties", {})
    if not props:
        errors.append(f"[{collection_name}] schema must have at least one field in 'properties'")
    if errors:
        return errors
    try:
        build_model_from_json_schema(schema_dict, model_name=collection_name)
    except Exception as exc:
        errors.append(f"[{collection_name}] invalid JSON Schema: {exc}")
    return errors


def _build_collection_to_pipeline_map(config_dict: dict) -> dict[str, str]:
    """Return a mapping from structured collection name to the pipeline that writes to it."""
    mapping: dict[str, str] = {}
    for pipeline in config_dict.get("pipelines", []):
        pipeline_name = pipeline.get("name", "")
        for step in pipeline.get("steps", []):
            coll = step.get("collection", "")
            if coll:
                mapping[coll] = pipeline_name
    return mapping


def _check_steps_cross_pipeline_doc_id(
    steps: list,
    driver_pipeline: str,
    coll_to_pipeline: dict[str, str],
    wf_name: str,
) -> list[str]:
    errors: list[str] = []
    for step in steps:
        inner = step.get("steps")
        if inner:
            errors.extend(
                _check_steps_cross_pipeline_doc_id(inner, driver_pipeline, coll_to_pipeline, wf_name)
            )
        if step.get("tool") != "structured-query":
            continue
        coll = step.get("collection", "")
        target_pipeline = coll_to_pipeline.get(coll)
        if not target_pipeline or target_pipeline == driver_pipeline:
            continue
        doc_id_val = str((step.get("filters") or {}).get("doc_id", ""))
        if "{{" in doc_id_val:
            errors.append(
                f"Workflow '{wf_name}' step '{step.get('id', '?')}': structured-query on "
                f"collection '{coll}' (pipeline '{target_pipeline}') filtered by doc_id "
                f"derived from pipeline '{driver_pipeline}'. Records in '{coll}' carry "
                f"'{target_pipeline}' doc_ids — this filter always returns zero results. "
                f"Use vector-search to retrieve cross-pipeline reference material instead."
            )
    return errors


def _validate_workflow_cross_pipeline_doc_id_filters(config_dict: dict) -> list[str]:
    """Detect structured-query steps filtering a cross-pipeline collection by a doc_id template."""
    coll_to_pipeline = _build_collection_to_pipeline_map(config_dict)
    errors: list[str] = []
    for workflow in config_dict.get("workflows", []):
        wf_name = workflow.get("name", "")
        pfc = workflow.get("params_from_collection") or {}
        driver_coll = pfc.get("collection", "")
        driver_pipeline = coll_to_pipeline.get(driver_coll)
        if not driver_pipeline:
            continue
        errors.extend(
            _check_steps_cross_pipeline_doc_id(
                workflow.get("steps", []),
                driver_pipeline,
                coll_to_pipeline,
                wf_name,
            )
        )
    return errors


def _parse_and_validate_schemas(
    raw: str,
    *,
    validator,
) -> tuple[dict | None, list[str]]:
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, [f"Schema YAML is not valid: {exc}"]
    if not isinstance(parsed, dict):
        return None, ["schemas_yaml must be a mapping of collection_name → JSON Schema object"]
    errors: list[str] = []
    for collection_name, schema_dict in parsed.items():
        errors.extend(validator(schema_dict, collection_name))
    return parsed, errors


async def propose_app_config(llm: LLMBase, messages: list, *, needs_workflow: bool):
    """Orchestrate the app config generation pipeline.

    Yields ``{"type": "token", "token": ...}`` progress events followed by a single
    ``{"type": "result", "generation_context": ..., "config_yaml": ...}`` event.
    Runs 2 steps for pipeline-only apps, 4 steps when ``needs_workflow`` is True.
    """
    yield {"type": "token", "token": "Generating extraction schemas...\n"}
    ext_output, extraction_schemas = await _run_propose_extraction_schemas(llm, messages)
    if extraction_schemas is None:
        yield {"type": "result", "generation_context": f"Extraction schema generation failed: {ext_output}", "config_yaml": None}
        return

    yield {"type": "token", "token": "Generating pipeline config...\n"}
    pipe_output, pipeline_config_dict, record_schemas, stored_yaml = (
        await _run_propose_pipeline_config(llm, messages, extraction_schemas)
    )
    if pipeline_config_dict is None:
        yield {"type": "result", "generation_context": f"Pipeline config generation failed: {pipe_output}", "config_yaml": None}
        return

    if not needs_workflow:
        yield {"type": "result", "generation_context": "Config generation complete.", "config_yaml": stored_yaml}
        return

    yield {"type": "token", "token": "Generating workflow schemas...\n"}
    wf_schema_output, workflow_schemas = await _run_propose_workflow_schemas(
        llm, messages, record_schemas
    )
    if workflow_schemas is None:
        yield {"type": "result", "generation_context": f"Workflow schema generation failed: {wf_schema_output}", "config_yaml": None}
        return

    yield {"type": "token", "token": "Generating workflow config...\n"}
    wf_config_output, wf_config_yaml = await _run_propose_workflow_config(
        llm, messages, pipeline_config_dict, record_schemas, workflow_schemas
    )
    if wf_config_yaml is None:
        yield {"type": "result", "generation_context": f"Workflow config generation failed: {wf_config_output}", "config_yaml": None}
    else:
        yield {"type": "result", "generation_context": "Config generation complete.", "config_yaml": wf_config_yaml}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _schemas_context(
    title: str,
    schemas: dict[str, str],
    *,
    intro: str | None = None,
) -> str:
    lines = [f"\n\n## {title}\n"]
    if intro:
        lines.append(intro)
    if not schemas:
        lines.append("{}")
        return "\n".join(lines)
    for coll_name, schema_json in schemas.items():
        lines.append(f"  {coll_name}: '{schema_json}'")
    return "\n".join(lines)


def _pipeline_config_context(pipeline_config_dict: dict) -> str:
    config_yaml = yaml.dump(
        pipeline_config_dict,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    return f"\n\n## Validated pipeline config\n\n{config_yaml}"


def _extract_record_schemas(pipeline_config_dict: dict) -> dict[str, str]:
    """Return full record schemas (extraction fields + doc_id [+ id_field]) keyed by collection name.

    Populated by _inject_pipeline_record_schemas after pipeline config validation.
    """
    return {
        sc["name"]: sc["schema"]
        for sc in pipeline_config_dict.get("structured_collections", [])
        if sc.get("name") and sc.get("schema")
    }


async def _run_propose_extraction_schemas(
    llm: LLMBase, conversation_messages: list
) -> tuple[str, dict[str, str] | None]:
    sub_messages = [{"role": "system", "content": _EXTRACTION_SCHEMA_AGENT_SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in conversation_messages
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

    errors: list[str] = []
    for attempt in range(_MAX_SCHEMA_RETRIES):
        result = await llm.complete(sub_messages, temperature=0.2)
        schemas_yaml = (result.get("content") or "").strip()
        schemas, errors = _parse_and_validate_schemas(
            schemas_yaml,
            validator=_validate_extraction_schema,
        )

        if not errors:
            logger.info(
                "generate/propose_extraction_schemas validated schemas=%s attempt=%d",
                list(schemas),
                attempt + 1,
            )
            schemas_as_json = {
                name: json.dumps(schema_dict, separators=(",", ":"))
                for name, schema_dict in schemas.items()
            }
            field_summary = "\n".join(
                f"  {name}: {', '.join(schema_dict.get('properties', {}).keys())}"
                for name, schema_dict in schemas.items()
            )
            return f"Schemas validated.\n{field_summary}", schemas_as_json

        logger.warning(
            "generate/propose_extraction_schemas attempt=%d errors=%s, schemas_yaml=%s",
            attempt + 1,
            errors,
            schemas_yaml,
        )
        error_text = "\n".join(f"- {e}" for e in errors)
        sub_messages += [
            {"role": "assistant", "content": schemas_yaml},
            {
                "role": "user",
                "content": f"Validation errors — fix and output the corrected YAML only:\n{error_text}",
            },
        ]

    return (
        f"Extraction schema generation failed after {_MAX_SCHEMA_RETRIES} attempts. Last errors:\n"
        + "\n".join(f"- {e}" for e in errors),
        None,
    )


async def _run_propose_workflow_schemas(
    llm: LLMBase,
    conversation_messages: list,
    record_schemas: dict[str, str],
) -> tuple[str, dict[str, str] | None]:
    # The tool description tells the model to call this only when the design has
    # workflow output collections, but we also accept an empty `{}` result as a
    # safety net so a misjudged call returns a clear instruction instead of an
    # error — the LLM is told to proceed to propose_workflow_config in that case.
    system_prompt = (
        _WORKFLOW_SCHEMA_AGENT_SYSTEM_PROMPT
        + _schemas_context(
            "Validated pipeline record schemas",
            record_schemas,
            intro=(
                "Full stored record schemas for pipeline-backed structured collections — "
                "use these to understand what fields and identifiers are available in each collection:"
            ),
        )
    )
    sub_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in conversation_messages
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

    errors: list[str] = []
    for attempt in range(_MAX_SCHEMA_RETRIES):
        result = await llm.complete(sub_messages, temperature=0.2)
        schemas_yaml = (result.get("content") or "").strip()
        schemas, errors = _parse_and_validate_schemas(
            schemas_yaml,
            validator=_validate_workflow_output_schema,
        )

        if not errors:
            logger.info(
                "generate/propose_workflow_schemas validated schemas=%s attempt=%d",
                list(schemas),
                attempt + 1,
            )
            schemas_as_json = {
                name: json.dumps(schema_dict, separators=(",", ":"))
                for name, schema_dict in schemas.items()
            }
            if not schemas_as_json:
                logger.info(
                    "generate/propose_workflow_schemas no workflow output collections attempt=%d",
                    attempt + 1,
                )
                return (
                    "No workflow output collections in this design. "
                    "Proceed to propose_workflow_config.",
                    schemas_as_json,
                )
            field_summary = "\n".join(
                f"  {name}: {', '.join(schema_dict.get('properties', {}).keys())}"
                for name, schema_dict in schemas.items()
            )
            return f"Workflow schemas validated.\n{field_summary}", schemas_as_json

        logger.warning(
            "generate/propose_workflow_schemas attempt=%d errors=%s schemas_yaml=%s",
            attempt + 1,
            errors,
            schemas_yaml,
        )
        error_text = "\n".join(f"- {e}" for e in errors)
        sub_messages += [
            {"role": "assistant", "content": schemas_yaml},
            {
                "role": "user",
                "content": f"Validation errors — fix and output the corrected YAML only:\n{error_text}",
            },
        ]

    return (
        f"Workflow schema generation failed after {_MAX_SCHEMA_RETRIES} attempts. Last errors:\n"
        + "\n".join(f"- {e}" for e in errors),
        None,
    )


async def _run_propose_pipeline_config(
    llm: LLMBase,
    conversation_messages: list,
    extraction_schemas: dict[str, str],
) -> tuple[str, dict | None, dict[str, str] | None, str | None]:
    """Generate and validate the pipeline section of the app config.

    Returns (tool_output, pipeline_config_dict, record_schemas, stored_yaml).
    stored_yaml is the serialized pipeline config; workflow sections are added in a later step.
    record_schemas contains the full stored record schemas (extraction fields + doc_id +
    id_field) for each pipeline-backed structured collection.
    """
    system_prompt = _PIPELINE_CONFIG_AGENT_SYSTEM_PROMPT + _schemas_context(
        "Validated pipeline extraction schemas",
        extraction_schemas,
        intro="Use these extraction_schema values verbatim for extract-structured steps:",
    )
    sub_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in conversation_messages
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

    errors: list[str] = []
    for attempt in range(_MAX_CONFIG_RETRIES):
        result = await llm.complete(sub_messages, temperature=0.2)
        config_yaml = (result.get("content") or "").strip()
        try:
            config_dict = yaml.safe_load(config_yaml)
            if not isinstance(config_dict, dict):
                raise ValueError("YAML must be a mapping at the top level")
            _inject_pipeline_record_schemas(config_dict)
            config = AppConfig.model_validate(config_dict)
        except Exception as exc:
            errors = [str(exc)]
            logger.warning(
                "generate/propose_pipeline_config attempt=%d errors=%s, config_yaml=%s",
                attempt + 1,
                errors,
                config_yaml,
            )
            error_text = "\n".join(f"- {e}" for e in errors)
            sub_messages += [
                {"role": "assistant", "content": config_yaml},
                {
                    "role": "user",
                    "content": f"Validation errors — fix and output the corrected YAML only:\n{error_text}",
                },
            ]
            continue

        record_schemas = _extract_record_schemas(config_dict)
        stored_yaml = config.to_yaml()
        logger.info(
            "generate/propose_pipeline_config validated app=%s attempt=%d record_schemas=%s",
            config.name,
            attempt + 1,
            list(record_schemas),
        )
        return "Pipeline config validated.", config_dict, record_schemas, stored_yaml

    return (
        f"Pipeline config generation failed after {_MAX_CONFIG_RETRIES} attempts. Last errors:\n"
        + "\n".join(f"- {e}" for e in errors),
        None,
        None,
        None,
    )


async def _run_propose_workflow_config(
    llm: LLMBase,
    conversation_messages: list,
    pipeline_config_dict: dict | None,
    record_schemas: dict[str, str],
    workflow_schemas: dict[str, str],
) -> tuple[str, str | None]:
    """Generate the workflow additions and assemble the final validated app config.

    The LLM generates only the workflow output structured_collections and workflows.
    These are merged with the already-validated pipeline_config_dict.
    """
    if not pipeline_config_dict:
        return "Pipeline config not available — call propose_pipeline_config first.", None

    workflow_schemas = workflow_schemas or {}
    system_prompt = (
        _WORKFLOW_CONFIG_AGENT_SYSTEM_PROMPT
        + _pipeline_config_context(pipeline_config_dict)
        + _schemas_context(
            "Validated pipeline record schemas",
            record_schemas,
            intro=(
                "Full stored record schemas (extraction fields + injected doc_id; "
                "RecordMode.MANY collections also include the per-record id_field):"
            ),
        )
        + _schemas_context(
            "Validated workflow output schemas",
            workflow_schemas,
            intro=(
                "Use these values verbatim for llm-structured output_schema and "
                "workflow output structured_collections schema:"
            ),
        )
    )
    sub_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in conversation_messages
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

    errors: list[str] = []
    for attempt in range(_MAX_CONFIG_RETRIES):
        result = await llm.complete(sub_messages, temperature=0.2)
        workflow_yaml = (result.get("content") or "").strip()
        try:
            workflow_additions = yaml.safe_load(workflow_yaml)
            if not isinstance(workflow_additions, dict):
                raise ValueError("YAML must be a mapping at the top level")
            merged_dict = copy.deepcopy(pipeline_config_dict)
            for sc in workflow_additions.get("structured_collections", []):
                merged_dict.setdefault("structured_collections", []).append(sc)
            merged_dict["workflows"] = workflow_additions.get("workflows", [])
            if not merged_dict["workflows"]:
                raise ValueError("workflows section is empty — at least one workflow is required")
            _inject_workflow_output_schemas(merged_dict, workflow_schemas)
            cross_pipeline_errors = _validate_workflow_cross_pipeline_doc_id_filters(merged_dict)
            if cross_pipeline_errors:
                raise ValueError("\n".join(cross_pipeline_errors))
            config = AppConfig.model_validate(merged_dict)
        except Exception as exc:
            errors = [str(exc)]
            logger.warning(
                "generate/propose_workflow_config attempt=%d errors=%s workflow_yaml=%s",
                attempt + 1,
                errors,
                workflow_yaml,
            )
            error_text = "\n".join(f"- {e}" for e in errors)
            sub_messages += [
                {"role": "assistant", "content": workflow_yaml},
                {
                    "role": "user",
                    "content": f"Validation errors — fix and output the corrected YAML only:\n{error_text}",
                },
            ]
            continue

        stored_yaml = config.to_yaml()
        logger.info(
            "generate/propose_workflow_config validated app=%s attempt=%d",
            config.name,
            attempt + 1,
        )
        return "Config validated.", stored_yaml

    return (
        f"Workflow config generation failed after {_MAX_CONFIG_RETRIES} attempts. Last errors:\n"
        + "\n".join(f"- {e}" for e in errors),
        None,
    )
