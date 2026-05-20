"""App generator endpoints — agentic, conversational config.yaml creation.

The LLM drives the conversation via schema/config tools. Schemas and configs
validate server-side and return errors for the LLM to fix. The client owns the
full message history (role: user/assistant only); tool call/result messages live
only within a single server turn.

Endpoints
---------
  POST /generate/chat    stateless chat turn; agent loop runs server-side
  POST /generate/deploy  create and activate an application from a config_yaml
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone

import yaml
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import AppCacheDep, SystemResourcesDep, SystemStoreDep
from api.factory import build_app
from api.models import (
    DeployResponse,
    GenerateChatRequest,
    GenerateChatResponse,
    GenerateDeployRequest,
)
from api.system_store import AppRecord
from cogbase.config.config import AppConfig
from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema
from cogbase.llms.base import LLMBase, ToolDefinition

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])

_MAX_AGENT_CALLS = 10

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_PROPOSE_EXTRACTION_SCHEMAS_TOOL: ToolDefinition = {
    "name": "propose_extraction_schemas",
    "description": (
        "Formalize the user-confirmed ingestion field list into validated JSON Schemas "
        "for structured collections produced by pipeline extract-structured steps only. "
        "Call this only after the user has confirmed the ingestion fields. "
        "Returns a brief validation summary on success, or a validation error message."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

_PROPOSE_WORKFLOW_SCHEMAS_TOOL: ToolDefinition = {
    "name": "propose_workflow_schemas",
    "description": (
        "Generate and validate JSON Schemas for workflow output collections used as "
        "llm-structured output_schema and structured-save storage schema. Call only if "
        "the confirmed design includes workflows that save structured records, and only "
        "after propose_extraction_schemas has succeeded."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

_PROPOSE_CONFIG_TOOL: ToolDefinition = {
    "name": "propose_app_config",
    "description": (
        "Generate and validate a complete CogBase app config YAML. "
        "Call this after propose_extraction_schemas has succeeded, and after "
        "propose_workflow_schemas too when the app has workflow output collections. "
        "The config is generated from the conversation and validated server-side. "
        "Returns 'Config validated.' on success, or a validation error message."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

_GENERATOR_TOOLS: list[ToolDefinition] = [
    _PROPOSE_EXTRACTION_SCHEMAS_TOOL,
    _PROPOSE_WORKFLOW_SCHEMAS_TOOL,
    _PROPOSE_CONFIG_TOOL,
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
a CogBase application and the validated pipeline extraction schemas below, produce \
JSON Schema definitions only for workflow output collections written by \
structured-save and used as llm-structured output_schema.

Generate schemas that match exactly the workflow output fields the user has \
already confirmed in the conversation — do not add, remove, or rename fields.

Workflow output schemas are not extraction schemas. They describe records created \
by workflow logic, so they may include stable identifiers and provenance fields \
such as doc_id, clause_id, finding_id, source record ids, status fields, evidence \
references, and LLM judgment fields.

Use the validated pipeline extraction schemas to preserve upstream identifiers and \
concepts. Pipeline storage records include the extracted fields plus injected doc_id. \
If a workflow iterates over an extracted collection, include the fields needed to \
trace each output record back to its source record or source document.

Output ONLY a YAML mapping of workflow_output_collection_name → JSON Schema object. \
If the confirmed design has no workflow output collections, output an empty YAML \
mapping: {}

Schema rules:
- Top-level keys are workflow output collection names (snake_case)
- Each non-empty collection must be type: object with a non-empty properties block
- doc_id is allowed when it is useful provenance
- Include a stable identifier field when the workflow creates independently saved records \
  (for example clause_id, finding_id, company_id, review_id)
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

_CONFIG_AGENT_SYSTEM_PROMPT = f"""\
You are a CogBase configuration generator. Given a conversation about building a CogBase \
application — including the validated pipeline extraction schemas and workflow output \
schemas injected below — produce a complete, valid config.yaml.

Output ONLY the raw YAML — no explanation, no markdown fences.

Use the extraction_schema values from the "Validated pipeline extraction schemas" section verbatim — \
do not rewrite or reformat them. For workflow output collections (structured-save targets), \
set schema inline using the exact value from "Validated workflow output schemas".

## Rules
1. name must be kebab-case (lowercase, alphanumeric, hyphens only)
2. chunk-embed-upsert is always the first pipeline step
3. Do NOT include doc_id in extraction schemas — it is injected automatically
4. For every extract-structured step, choose record_mode based on how many records the LLM \
   should return per document:
   - record_mode: one (default) — the LLM returns ONE record for the whole document. \
     Use when the document yields exactly one entity of this type. Examples: contract-level \
     metadata (one set of facts per contract), board update KPIs (one snapshot per deck), \
     deal memo summary (one memo per document).
   - record_mode: many — the LLM returns MULTIPLE records as a list. Use when the document \
     contains a list of distinct entities of this type. Examples: clauses in a contract, \
     rules/prohibitions/requirements in a policy document, line items in an invoice, \
     findings in an audit report, employees in a roster, transactions in a statement. \
     When record_mode is many, ALSO set: response_field (the array key in the extractor \
     output, e.g. clauses, rules, items), id_field (the per-record identifier, e.g. \
     clause_id, rule_id, item_id), and id_template (e.g. "{{doc_id}}__{{index:04d}}").
   Heuristic: if a natural plural ("the clauses", "the rules", "the line items") describes \
   what the schema captures, choose record_mode: many. If the schema is a flat set of \
   header-level facts about the document itself, choose record_mode: one. Getting this wrong \
   collapses many entities into a single record (loses data) or wraps a singleton in a list \
   for no reason.
5. All content is INLINE — do not use .json or .txt filenames as values anywhere
6. Pipeline step collections must exactly match declared vector/structured collection names
7. Use snake_case for all collection names and field names
8. Every pipeline must have a routing_description — a plain-language sentence describing which documents belong in that pipeline (used by LLM routing to classify documents)
9. output_schema in llm-structured workflow steps must be an inline JSON string — use the \
   exact value from "Validated workflow output schemas". Never use a .json filename.
10. prompt in llm-structured workflow steps must be inline text. Never use a .txt filename.
11. Workflow output collections (structured-save targets not produced by extract-structured) \
    must have schema set inline using the value from "Validated workflow output schemas" — they \
    are NOT auto-injected like pipeline collections.
12. Every workflow must have a params_from_collection block that derives input params from a \
    structured collection. Use filters to select by doc_id and params to expose the values \
    the workflow steps reference via {{{{ input.* }}}}.
13. Avoid bulk context in judgment workflows:
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

## Config format

{AppConfig.config_format_prompt()}

## Example — two document types, shared vector collections

name: vc-portfolio

vector_collections:
  - name: portfolio_chunks
    description: Full-text passages from all portfolio documents.
  - name: portfolio_summaries
    description: One-per-document summaries of portfolio documents.

structured_collections:
  - name: portfolio_kpis
    description: >
      Financial and operational KPIs extracted from board decks and LP updates.
      Use for exact lookups and cross-company comparisons.

pipelines:
  # board-update pipeline: chunk + extract KPIs + summarize
  - name: board-update
    routing_description: Board decks and LP updates from portfolio companies containing financial KPIs, headcount, milestones, and risk sections.
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
    routing_description: Investment memos and pitch decks for prospective deals containing investment thesis, deal terms, and sector analysis.
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

## Example — workflow: two pipelines, mixed record_mode, fan-out LLM judgment

# Demonstrates: two pipelines routed by metadata; one extract-structured step with
# record_mode: one (document-level metadata, one record per contract) alongside another
# with record_mode: many (clauses, many records per contract); a workflow that fans out
# over the many-records collection and saves judgments to a workflow output collection.

name: contract-compliance

vector_collections:
  - name: rule_chunks
    description: >
      Company policy passages and standards used as evidence for compliance judgments.
  - name: contract_chunks
    description: >
      Contract text passages for detailed questions about specific contract terms or clauses.

structured_collections:
  - name: contract_metadata
    description: >
      Key facts per contract: parties, dates, value, governing law, termination
      notice period. ONE record per contract document.
  - name: contract_clauses
    description: >
      Individual clauses extracted from contracts. MANY records per contract — each
      record is one clause with its type and verbatim text. Filter by doc_id to
      retrieve all clauses for a contract, or by clause_type for a specific category.
  - name: clause_compliance_findings
    description: >
      Clause-level compliance findings produced by the compliance workflow. Each record
      captures whether a clause complies with company policy, with severity and reasoning.

pipelines:
  # rules pipeline: company policy documents — index only, no structured extraction needed
  - name: rules
    routing_description: Company policy documents, internal standards, compliance guidelines, and fallback positions that define rules contracts must be checked against.
    match:
      metadata:
        doc_type: rules
    steps:
      - tool: chunk-embed-upsert
        collection: rule_chunks

  # contracts pipeline: chunk + extract document-level metadata + extract clauses
  - name: contracts
    routing_description: Vendor contracts, commercial agreements, and service agreements to be reviewed for compliance against company policy.
    match:
      metadata:
        doc_type: contract
    steps:
      - tool: chunk-embed-upsert
        collection: contract_chunks

      # ONE-PER-DOCUMENT extraction — record_mode: one (default, omitted). The contract has
      # exactly one set of header-level facts, so the LLM returns a single JSON object.
      - tool: extract-structured
        collection: contract_metadata
        extractor:
          type: llm
          extraction_schema: '{{"type":"object","properties":{{"contract_type":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Contract category"}},"parties":{{"type":"array","items":{{"type":"object","properties":{{"name":{{"type":"string"}},"role":{{"type":"string"}}}}}},"description":"Named parties and their roles"}},"effective_date":...,"expiry_date":...,"contract_value":...,"governing_law":...,"termination_notice_days":...}}}}'
          prompt: |
            You are a legal contract analyst. Extract key contract-level facts from the
            contract provided.

            Rules:
            - Use null for any field not present in the document. Do not invent.
            - Format dates as YYYY-MM-DD.
            - For parties, return an array of {{name, role}} objects.
            - Return ONLY the JSON object — no explanation, no markdown fences.

      # MANY-PER-DOCUMENT extraction — record_mode: many. A contract contains a list of
      # distinct clauses, so the LLM returns an array under response_field, and each
      # element becomes one record keyed by id_field.
      - tool: extract-structured
        collection: contract_clauses
        extractor:
          type: llm
          extraction_schema: '{{"type":"object","properties":{{"clause_type":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Clause category: liability, indemnification, termination, payment, privacy, confidentiality, ip, governing_law, other"}},"text":{{"type":"string","description":"Verbatim clause text"}}}}}}'
          record_mode: many
          response_field: clauses
          id_field: clause_id
          id_template: "{{doc_id}}__{{index:04d}}"
          prompt: |
            You are a legal contract analyst. Extract every distinct clause from the
            contract provided.

            Rules:
            - Copy all clause text verbatim — do not paraphrase.
            - Assign clause_type from: liability, indemnification, termination, payment,
              privacy, confidentiality, ip, governing_law, other. Use null when unclear.
            - Return ONLY the JSON object — no explanation, no markdown fences.

workflows:
  - name: check-contract-compliance
    trigger:
      type: manual
    # Derives doc_id from contract_metadata (the one-per-document collection) so the
    # workflow runs once per contract, even though it then fans out over many clauses.
    params_from_collection:
      collection: contract_metadata
      filters:
        doc_id: "{{{{ doc.doc_id }}}}"
      params:
        doc_id: "{{{{ record.doc_id }}}}"
    steps:
      - id: load_clauses
        tool: structured-query
        collection: contract_clauses
        filters:
          doc_id: "{{{{ input.doc_id }}}}"

      - id: review_each_clause
        foreach: "{{{{ steps.load_clauses.records }}}}"
        steps:
          # Retrieve only the rules relevant to this clause — avoids loading the entire
          # rules collection (see rule 13).
          - id: retrieve_rules
            tool: vector-search
            collection: rule_chunks
            query: "{{{{ item.clause_type }}}}\\n{{{{ item.text }}}}"
            top_k: 5

          - id: judge
            tool: llm-structured
            prompt: |
              You are a contract compliance reviewer. Judge whether the clause complies
              with company policy using ONLY the policy excerpts provided.

              Rules:
              - Ground every finding exclusively in the provided excerpts.
              - If excerpts are insufficient, set status=needs_review.
              - Return ONLY valid JSON — no markdown fences, no explanation.
            input:
              clause: "{{{{ item }}}}"
              rules: "{{{{ steps.retrieve_rules.chunks }}}}"
            output_schema: '{{"type":"object","properties":{{"clause_id":{{"type":"string"}},"doc_id":{{"type":"string"}},"status":{{"type":"string","enum":["compliant","non_compliant","needs_review","not_applicable"]}},"severity":{{"type":"string","enum":["low","medium","high","critical"]}},"summary":{{"type":"string"}},"reasoning":{{"type":"string"}}}}}}'

          - id: save_finding
            tool: structured-save
            collection: clause_compliance_findings
            records:
              - "{{{{ steps.judge.output }}}}" """

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
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
   as a bullet list. Use nested bullets for object or array fields. For example:

   **contracts**
   - vendor_name — name of the vendor
   - effective_date — contract start date (ISO 8601)
   - payment_terms
     - schedule - payment schedule, e.g. "net-30", "monthly", "upfront", "milestone-based"
     - late_penalty - penalty or interest rate for late payment, verbatim if present
   - key_terms (list) - significant defined terms, unusual provisions

   Use domain knowledge to propose sensible fields — do not ask the user to enumerate them.
   Ask the user to confirm, add, remove, or rename fields. Revise conversationally until confirmed.

   If any queries require analytical fan-out (e.g. "check every clause for compliance",
   "rank all companies by ARR"), also design the workflow before calling propose_extraction_schemas:
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

   Once the field list (and any workflow design) is confirmed, call propose_extraction_schemas.
   If the confirmed design includes workflow output collections, then call propose_workflow_schemas.
   When the needed schema tools succeed, immediately call propose_app_config — no additional confirmation needed.

3. Once the schema is confirmed, call propose_app_config.
   It generates and validates the full config — pipelines and any confirmed workflows — from
   the conversation. When it succeeds, present the result to the user with a plain-language
   explanation of what was set up and why."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            except (json.JSONDecodeError, ValueError):
                continue
            record_id_field = (
                extractor.get("id_field") if extractor.get("record_mode") == "many" else None
            )
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
    if workflow_schemas:
        save_targets: set[str] = set()
        for workflow in config_dict.get("workflows", []):
            _collect_save_targets(workflow.get("steps", []), save_targets)
        for sc in config_dict.get("structured_collections", []):
            name = sc.get("name")
            if name in save_targets and name in workflow_schemas:
                try:
                    schema_dict = json.loads(workflow_schemas[name])
                    sc["schema"] = json.dumps(schema_dict, separators=(",", ":"))
                except (json.JSONDecodeError, ValueError):
                    pass


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


def _serialize_config(config: AppConfig) -> str:
    return yaml.dump(
        config.model_dump(by_alias=True, mode="json", exclude_none=True),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


async def _chat_turn_events(
    body: GenerateChatRequest,
    system_resources: SystemResourcesDep,
    *,
    log_prefix: str,
):
    llm = system_resources.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM configured on the system")

    from cogbase.llms.base import ChatMessage as LLMChatMessage

    logger.info("%s start text=%s ..., history=%d", log_prefix, body.text[:50], len(body.history))

    messages: list[LLMChatMessage] = (
        [{"role": "system", "content": _SYSTEM_PROMPT}]
        + [{"role": m.role, "content": m.content} for m in body.history]
        + [{"role": "user", "content": body.text}]
    )

    validated_config_yaml: str | None = None
    extraction_schemas: dict[str, str] = {}
    workflow_schemas: dict[str, str] = {}
    final_content: str = ""
    result = None

    try:
        for call_num in range(_MAX_AGENT_CALLS):
            streamed_chunks: list[str] = []
            result = None
            async for chunk in llm.complete_stream(messages, tools=_GENERATOR_TOOLS, temperature=0.3):
                if isinstance(chunk, str):
                    streamed_chunks.append(chunk)
                    yield {"type": "token", "token": chunk}
                else:
                    result = chunk  # CompletionResult carrying tool_calls

            tool_calls = result.get("tool_calls") if result else None

            if not tool_calls:
                final_content = "".join(streamed_chunks).strip()
                break

            messages.append({
                "role": "assistant",
                "content": result.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            })

            tool_names = ", ".join(tc["name"] for tc in tool_calls)
            logger.info("%s call=%d tools=%s", log_prefix, call_num + 1, tool_names)

            for tc in tool_calls:
                if tc["name"] == "propose_extraction_schemas":
                    yield {"type": "token", "token": "Generating extraction schemas...\n"}
                    tool_output, schemas = await _run_propose_extraction_schemas(llm, messages)
                    if schemas is not None:
                        extraction_schemas = schemas
                elif tc["name"] == "propose_workflow_schemas":
                    yield {"type": "token", "token": "Generating workflow schemas...\n"}
                    tool_output, schemas = await _run_propose_workflow_schemas(
                        llm,
                        messages,
                        extraction_schemas,
                    )
                    if schemas is not None:
                        workflow_schemas = schemas
                elif tc["name"] == "propose_app_config":
                    yield {"type": "token", "token": "Generating app config...\n"}
                    tool_output, config_yaml = await _run_propose_config(
                        llm,
                        messages,
                        extraction_schemas,
                        workflow_schemas,
                    )
                    if config_yaml is not None:
                        validated_config_yaml = config_yaml
                else:
                    tool_output = f"Unknown tool: {tc['name']}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })
        else:
            logger.warning(
                "%s reached max_calls=%d without final answer",
                log_prefix,
                _MAX_AGENT_CALLS,
            )
            final_content = "".join(streamed_chunks).strip()

        logger.info(
            "%s turn=%d config_validated=%s final_content=%d",
            log_prefix,
            len(body.history) + 1,
            validated_config_yaml is not None,
            len(final_content),
        )
        yield {
            "type": "result",
            "result": {"content": final_content, "config_yaml": validated_config_yaml},
        }
    except Exception:
        logger.exception("%s failed", log_prefix)
        yield {"type": "error", "error": "stream failed"}


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
    extraction_schemas: dict[str, str],
) -> tuple[str, dict[str, str] | None]:
    # The tool description tells the model to call this only when the design has
    # workflow output collections, but we also accept an empty `{}` result as a
    # safety net so a misjudged call returns a clear instruction instead of an
    # error — the LLM is told to proceed to propose_app_config in that case.
    system_prompt = (
        _WORKFLOW_SCHEMA_AGENT_SYSTEM_PROMPT
        + _schemas_context("Validated pipeline extraction schemas", extraction_schemas)
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
                    "Proceed to propose_app_config.",
                    schemas_as_json,
                )
            field_summary = "\n".join(
                f"  {name}: {', '.join(schema_dict.get('properties', {}).keys())}"
                for name, schema_dict in schemas.items()
            )
            return f"Workflow schemas validated.\n{field_summary}", schemas_as_json

        logger.warning(
            "generate/propose_workflow_schemas attempt=%d errors=%s", attempt + 1, errors
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


async def _run_propose_config(
    llm: LLMBase,
    conversation_messages: list,
    extraction_schemas: dict[str, str],
    workflow_schemas: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    workflow_schemas = workflow_schemas or {}
    schema_context = _schemas_context(
        "Validated pipeline extraction schemas",
        extraction_schemas,
        intro="Use these extraction_schema values verbatim for extract-structured steps:",
    ) + _schemas_context(
        "Validated workflow output schemas",
        workflow_schemas,
        intro=(
            "Use these values verbatim for llm-structured output_schema and "
            "structured-save target collection schema:"
        ),
    )
    system_prompt = _CONFIG_AGENT_SYSTEM_PROMPT + schema_context

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
            _inject_workflow_output_schemas(config_dict, workflow_schemas)
            config = AppConfig.model_validate(config_dict)
        except Exception as exc:
            errors = [str(exc)]
            logger.warning(
                "generate/propose_config attempt=%d errors=%s", attempt + 1, errors
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

        stored_yaml = _serialize_config(config)
        logger.info(
            "generate/propose_config validated app=%s attempt=%d", config.name, attempt + 1
        )
        return "Config validated.", stored_yaml

    return (
        f"Config generation failed after {_MAX_CONFIG_RETRIES} attempts. Last errors:\n"
        + "\n".join(f"- {e}" for e in errors),
        None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=GenerateChatResponse)
async def chat(
    body: GenerateChatRequest,
    system_resources: SystemResourcesDep,
) -> GenerateChatResponse:
    """One stateless chat turn.

    The client maintains the full message history (role: user/assistant) and sends
    it each call. The agent loop runs entirely server-side: the LLM calls tools,
    gets results, and may call tools again — the client sees only the final response.
    """
    validated_config_yaml: str | None = None
    final_content: str = ""
    async for event in _chat_turn_events(body, system_resources, log_prefix="generate/chat"):
        if event["type"] == "result":
            result = event["result"]
            final_content = result["content"]
            validated_config_yaml = result["config_yaml"]
        elif event["type"] == "error":
            raise HTTPException(status_code=500, detail=event["error"])

    logger.info(
        "generate/chat turn=%d config_validated=%s, final_content=%d, %s ...",
        len(body.history) + 1,
        validated_config_yaml is not None,
        len(final_content),
        final_content[:50],
    )
    return GenerateChatResponse(
        content=final_content,
        config_yaml=validated_config_yaml,
    )


@router.post("/chat/stream")
async def chat_stream(
    body: GenerateChatRequest,
    system_resources: SystemResourcesDep,
) -> StreamingResponse:
    """Stream a generate chat turn as Server-Sent Events.

    Token events:  ``{"token": "<text>"}``
    Final event:   ``{"result": {"content": "...", "config_yaml": "..."}}``
    Sentinel:      ``data: [DONE]``
    """
    async def event_stream():
        try:
            async for event in _chat_turn_events(
                body,
                system_resources,
                log_prefix="generate/chat/stream",
            ):
                if event["type"] == "token":
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
                elif event["type"] == "result":
                    yield f"data: {json.dumps({'result': event['result']})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': event['error']})}\n\n"
        except Exception:
            logger.exception("generate/chat/stream failed")
            yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/deploy", response_model=DeployResponse, status_code=status.HTTP_201_CREATED)
async def deploy(
    body: GenerateDeployRequest,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    system_resources: SystemResourcesDep,
) -> DeployResponse:
    """Create and activate an application from a generated config_yaml."""
    try:
        config = AppConfig.from_yaml(body.config_yaml)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config: {exc}") from exc

    if await system_store.get_app(config.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application '{config.name}' already exists",
        )

    stored_yaml = _serialize_config(config)
    now = _now()
    record = AppRecord(
        name=config.name,
        config_yaml=stored_yaml,
        status="initializing",
        created_at=now,
        updated_at=now,
    )
    await system_store.save_app(record)

    try:
        app = await build_app(config, system=system_resources, app_status=record.status)
        app_cache.add(config.name, app)
        record = record.model_copy(update={"status": "active", "updated_at": _now()})
        logger.info("deployed app name=%s", config.name)
    except Exception as exc:
        logger.exception("deploy failed app=%s", config.name)
        record = record.model_copy(
            update={"status": "error", "error": str(exc), "updated_at": _now()}
        )

    await system_store.save_app(record)
    return DeployResponse(name=record.name, status=record.status, error=record.error)
