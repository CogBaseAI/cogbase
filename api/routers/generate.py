"""App generator endpoints — agentic, conversational config.yaml creation.

The LLM drives the conversation via two tools: propose_extraction_schema and
propose_app_config. Both validate server-side and return errors for the LLM
to fix. The client owns the full message history (role: user/assistant only);
tool call/result messages live only within a single server turn.

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

_PROPOSE_SCHEMA_TOOL: ToolDefinition = {
    "name": "propose_extraction_schema",
    "description": (
        "Formalize the user-confirmed field list into validated JSON Schemas for all "
        "structured collections. Call this only after the user has confirmed the field "
        "list in the conversation — the schemas are derived from those confirmed fields. "
        "Returns a brief validation summary on success, or a validation error message."
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
        "Call this after propose_extraction_schema has succeeded. "
        "The config is generated from the conversation and validated server-side. "
        "Returns 'Config validated.' on success, or a validation error message."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

_GENERATOR_TOOLS: list[ToolDefinition] = [_PROPOSE_SCHEMA_TOOL, _PROPOSE_CONFIG_TOOL]

_SCHEMA_AGENT_SYSTEM_PROMPT = """\
You are a CogBase schema designer. Given a conversation about building a CogBase \
application, produce JSON Schema definitions for every structured collection \
the application needs. Generate schemas that match exactly the fields the user \
has already confirmed in the conversation — do not add, remove, or rename fields.

CogBase has three store types — design schemas only for structured collections:
- Structured collections: discrete extractable facts for filtered/exact lookup (what you design here)
- Vector/chunk collections: full-text passages for semantic search (handled automatically by chunk-embed-upsert)
- Document collections: LLM summaries for high-level queries (handled automatically by document-embed-upsert)
Do NOT include fields like document_text, full_text, body, or summary — those are covered \
by the other two collections automatically.

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
- If the application has analytical workflows that call llm-structured and save results via \
structured-save, design schemas for those output collections too. They serve as both the \
llm-structured output_schema and the collection storage schema. Use an appropriate record \
identifier field (e.g. clause_id, finding_id) rather than doc_id.

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

_MAX_SCHEMA_RETRIES = 3
_MAX_CONFIG_RETRIES = 3

_CONFIG_AGENT_SYSTEM_PROMPT = f"""\
You are a CogBase configuration generator. Given a conversation about building a CogBase \
application — including the validated extraction schemas injected below — produce \
a complete, valid config.yaml.

Output ONLY the raw YAML — no explanation, no markdown fences.

Use the extraction_schema values from the "Validated extraction schemas" section verbatim — \
do not rewrite or reformat them. For pipeline collections (extract-structured targets), the \
schema field in structured_collections is derived automatically — you do not need to provide it. \
For workflow output collections (structured-save targets that are not extract-structured targets), \
set schema inline using the exact value from "Validated extraction schemas".

## Rules
1. name must be kebab-case (lowercase, alphanumeric, hyphens only)
2. chunk-embed-upsert is always the first pipeline step
3. Do NOT include doc_id in extraction schemas — it is injected automatically
4. All content is INLINE — do not use .json or .txt filenames as values anywhere
5. Pipeline step collections must exactly match declared vector/structured collection names
6. Use snake_case for all collection names and field names
7. Every pipeline must have a routing_description — a plain-language sentence describing which documents belong in that pipeline (used by LLM routing to classify documents)
8. output_schema in llm-structured workflow steps must be an inline JSON string — use the \
   exact value from "Validated extraction schemas". Never use a .json filename.
9. prompt in llm-structured workflow steps must be inline text. Never use a .txt filename.
10. Workflow output collections (structured-save targets not produced by extract-structured) \
    must have schema set inline using the value from "Validated extraction schemas" — they \
    are NOT auto-injected like pipeline collections.

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
    schema: ...
    primary_fields: [doc_id]

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
        chunker:
          type: langchain

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
        chunker:
          type: langchain

      - tool: document-embed-upsert
        collection: portfolio_summaries
        doc_prompt: >
          Summarize this investment memo or pitch deck. Include: company name,
          stage, sector, investment thesis, key risks, and deal terms if present.

## Example — workflow: fan-out LLM judgment over records in a collection

structured_collections:
  - name: clause_compliance_findings
    description: >
      Clause-level compliance findings. Each record captures whether a clause complies
      with company policy, with severity and plain-language assessment.
    schema: '{{"type":"object","properties":{{"clause_id":{{"type":"string","description":"Clause identifier"}},"compliant":{{"anyOf":[{{"type":"boolean"}},{{"type":"null"}}],"description":"Whether the clause complies"}},"severity":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Severity if non-compliant: critical/high/medium/low"}},"summary":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Plain-language compliance assessment"}}}}}}'
    primary_fields: [clause_id]

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
          doc_id: "{{{{ input.doc_id }}}}"

      - id: review_each_clause
        foreach: "{{{{ steps.load_clauses.records }}}}"
        steps:
          - id: retrieve_rules
            tool: vector-search
            collection: rule_chunks
            query: "{{{{ item.clause_type }}}}\\n{{{{ item.text }}}}"
            top_k: 5

          - id: judge
            tool: llm-structured
            prompt: |
              You are a compliance judge. Review the clause against the company policy rules
              provided. Return JSON only — no explanation, no markdown fences.
            input:
              clause: "{{{{ item }}}}"
              rules: "{{{{ steps.retrieve_rules.chunks }}}}"
            output_schema: '{{"type":"object","properties":{{"clause_id":{{"type":"string","description":"Clause identifier"}},"compliant":{{"anyOf":[{{"type":"boolean"}},{{"type":"null"}}],"description":"Whether the clause complies"}},"severity":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Severity if non-compliant: critical/high/medium/low"}},"summary":{{"anyOf":[{{"type":"string"}},{{"type":"null"}}],"description":"Plain-language compliance assessment"}}}}}}'

          - id: save_finding
            tool: structured-save
            collection: clause_compliance_findings
            records:
              - "{{{{ steps.judge.output }}}}" """

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
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
   "rank all companies by ARR"), also design the workflow before calling propose_extraction_schema:
   - Sketch the step sequence: what records to load, what to iterate over, what LLM judgment
     to apply, and where to save results.
   - Identify the workflow output collection (e.g. "compliance_findings") and add its fields
     to the proposed field list — its schema must be generated in the same call as the pipeline schemas.
   - Confirm the workflow design with the user.

   Once the field list (and any workflow design) is confirmed, call propose_extraction_schema.
   When it succeeds, immediately call propose_app_config — no additional confirmation needed.

3. Once the schema is confirmed, call propose_app_config.
   It generates and validates the full config — pipelines and any confirmed workflows — from
   the conversation. When it succeeds, present the result to the user with a plain-language
   explanation of what was set up and why."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_record_schema(extraction_schema: dict) -> dict:
    """Add a required doc_id string field to produce the record schema."""
    record = copy.deepcopy(extraction_schema)
    record.setdefault("properties", {})["doc_id"] = {
        "type": "string",
        "description": "document identifier",
    }
    required = record.setdefault("required", [])
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


def _inject_record_schemas(config_dict: dict, extracted_schemas: dict[str, str] | None = None) -> None:
    """Set schema for structured collections.

    Pipeline extract-structured targets: schema = extraction_schema + doc_id.
    Workflow structured-save targets: schema from extracted_schemas as-is (no doc_id injection).
    """
    ext_schemas: dict[str, dict] = {}
    for pipeline in config_dict.get("pipelines", []):
        for step in pipeline.get("steps", []):
            if step.get("tool") == "extract-structured":
                collection = step.get("collection")
                ext_schema_str = (step.get("extractor") or {}).get("extraction_schema", "")
                if collection and ext_schema_str:
                    try:
                        ext_schemas[collection] = json.loads(ext_schema_str)
                    except (json.JSONDecodeError, ValueError):
                        pass
    for sc in config_dict.get("structured_collections", []):
        name = sc.get("name")
        if name in ext_schemas:
            sc["schema"] = json.dumps(
                _make_record_schema(ext_schemas[name]), separators=(",", ":")
            )

    if extracted_schemas:
        save_targets: set[str] = set()
        for workflow in config_dict.get("workflows", []):
            _collect_save_targets(workflow.get("steps", []), save_targets)
        for sc in config_dict.get("structured_collections", []):
            name = sc.get("name")
            if name in save_targets and "schema" not in sc and name in extracted_schemas:
                try:
                    schema_dict = json.loads(extracted_schemas[name])
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


def _parse_and_validate_schemas(raw: str) -> tuple[dict | None, list[str]]:
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, [f"Schema YAML is not valid: {exc}"]
    if not isinstance(parsed, dict):
        return None, ["schemas_yaml must be a mapping of collection_name → JSON Schema object"]
    errors: list[str] = []
    for collection_name, schema_dict in parsed.items():
        errors.extend(_validate_extraction_schema(schema_dict, collection_name))
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
    extracted_schemas: dict[str, str] = {}
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
                if tc["name"] == "propose_extraction_schema":
                    yield {"type": "token", "token": "Generating extraction schema...\n"}
                    tool_output, schemas = await _run_propose_schema(llm, messages)
                    if schemas is not None:
                        extracted_schemas = schemas
                elif tc["name"] == "propose_app_config":
                    yield {"type": "token", "token": "Generating app config...\n"}
                    tool_output, config_yaml = await _run_propose_config(
                        llm, messages, extracted_schemas
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


async def _run_propose_schema(
    llm: LLMBase, conversation_messages: list
) -> tuple[str, dict[str, str] | None]:
    sub_messages = [{"role": "system", "content": _SCHEMA_AGENT_SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in conversation_messages
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

    errors: list[str] = []
    for attempt in range(_MAX_SCHEMA_RETRIES):
        result = await llm.complete(sub_messages, temperature=0.2)
        schemas_yaml = (result.get("content") or "").strip()
        schemas, errors = _parse_and_validate_schemas(schemas_yaml)

        if not errors:
            logger.info(
                "generate/propose_schema validated schemas=%s attempt=%d",
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
            "generate/propose_schema attempt=%d errors=%s", attempt + 1, errors
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
        f"Schema generation failed after {_MAX_SCHEMA_RETRIES} attempts. Last errors:\n"
        + "\n".join(f"- {e}" for e in errors),
        None,
    )


async def _run_propose_config(
    llm: LLMBase,
    conversation_messages: list,
    extraction_schemas: dict[str, str],
) -> tuple[str, str | None]:
    schema_lines = ["\n\n## Validated extraction schemas\n\nUse these extraction_schema values verbatim:"]
    for coll_name, schema_json in extraction_schemas.items():
        schema_lines.append(f"  {coll_name}: '{schema_json}'")
    system_prompt = _CONFIG_AGENT_SYSTEM_PROMPT + "\n".join(schema_lines)

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
            _inject_record_schemas(config_dict, extraction_schemas)
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
