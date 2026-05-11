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
        "Generate and validate extraction schemas for all structured collections. "
        "Call this once you understand the domain and document types — do not ask the "
        "user to enumerate fields. The schemas are derived from the conversation and "
        "domain knowledge, then presented to the user for review."
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
application, propose complete JSON Schema definitions for every structured collection \
the application needs.

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
application — including resolved extraction schemas from propose_extraction_schema — produce \
a complete, valid config.yaml.

Output ONLY the raw YAML — no explanation, no markdown fences.

Copy the extraction_schema JSON string from the conversation verbatim; \
do not rewrite or reformat it. The schema field in structured_collections is derived \
automatically from the extraction_schema — you do not need to provide it.

## Rules
1. name must be kebab-case (lowercase, alphanumeric, hyphens only)
2. chunk-embed-upsert is always the first pipeline step
3. Do NOT include doc_id in extraction schemas — it is injected automatically
4. All content is INLINE — do not use .json or .txt filenames as values anywhere
5. Pipeline step collections must exactly match declared vector/structured collection names
6. Use snake_case for all collection names and field names

## Config format

{AppConfig.config_format_prompt()}

## Example — two document types, shared vector collections, no workflows

name: vc-portfolio

vector_collections:
  - name: portfolio_chunks
    description: >
      Full-text passage index across all portfolio documents. Use for detailed
      questions about specific companies, deals, or periods.
  - name: portfolio_summaries
    description: >
      One-per-document summaries. Use for broad questions like "which companies
      raised a Series B?" or "which board decks mention hiring risk?"

structured_collections:
  - name: portfolio_kpis
    description: >
      Extracted KPIs from board decks and LP updates: company_name,
      reporting_period, arr_usd, burn_rate_monthly_usd, runway_months,
      headcount, ndr_pct, key_milestones, notable_risks. Use for exact
      lookups and cross-company comparisons.
    primary_fields: [doc_id]

pipelines:
  # board-update pipeline: chunk + extract KPIs + summarize
  - name: board-update
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
          prompt: >
            You are a VC portfolio analyst. Extract structured KPIs from the
            board deck or LP update. Use null for any field not present.
            Return ONLY the JSON object.

      - tool: document-embed-upsert
        collection: portfolio_summaries
        doc_prompt: >
          Summarize this board deck or LP update. Include: company name,
          reporting period, key financials, headcount, milestones, and risks.

  # deal-memo pipeline: chunk + summarize only — no KPI extraction from pitch decks
  - name: deal-memo
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
          stage, sector, investment thesis, key risks, and deal terms if present."""

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

**Multiple pipelines** — an app can declare multiple named pipelines, each matched to documents \
by metadata (e.g. `doc_type: board_update` vs `doc_type: deal_memo`). Pipelines can share \
vector collections; only the steps differ. Ask whether the user has multiple document types \
that need different treatment.

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

2. Once you understand the domain, call propose_extraction_schema.
   It uses domain expertise to propose fields and schemas for all structured collections —
   you do not enumerate fields or decide collection structure yourself.
   When it succeeds, present the proposed schemas to the user and ask for confirmation
   or corrections. Include the resolved JSON strings in your response so they persist
   in conversation history.

3. Once the schema is confirmed, call propose_app_config.
   It generates and validates the config from the conversation — you do not write the YAML yourself.
   When it succeeds, present the result to the user with a plain-language explanation of what was set up and why."""

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


def _inject_record_schemas(config_dict: dict) -> None:
    """Set schema = extraction_schema + doc_id for each extract-structured collection."""
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
        config.model_dump(by_alias=True, mode="json"),
        allow_unicode=True,
        default_flow_style=False,
    )


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _run_propose_schema(llm: LLMBase, conversation_messages: list) -> str:
    sub_messages = [{"role": "system", "content": _SCHEMA_AGENT_SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in conversation_messages
        if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
    ]

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
            lines = ["Schema validated. Use these values verbatim in your config:"]
            for name, schema_dict in schemas.items():
                ext_json = json.dumps(schema_dict, separators=(",", ":"))
                lines.append(f"\n{name}:")
                lines.append(f"  extraction_schema: '{ext_json}'")
            return "\n".join(lines)

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

    return f"Schema generation failed after {_MAX_SCHEMA_RETRIES} attempts. Last errors:\n" + "\n".join(
        f"- {e}" for e in errors  # type: ignore[possibly-undefined]
    )


async def _run_propose_config(llm: LLMBase, conversation_messages: list) -> tuple[str, str | None]:
    sub_messages = [{"role": "system", "content": _CONFIG_AGENT_SYSTEM_PROMPT}] + [
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
            _inject_record_schemas(config_dict)
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
    llm = system_resources.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM configured on the system")

    from cogbase.llms.base import ChatMessage as LLMChatMessage

    logger.info("generate/chat start text=%s ..., history=%d", body.text[:50], len(body.history))

    messages: list[LLMChatMessage] = (
        [{"role": "system", "content": _SYSTEM_PROMPT}]
        + [{"role": m.role, "content": m.content} for m in body.history]
        + [{"role": "user", "content": body.text}]
    )

    validated_config_yaml: str | None = None
    final_content: str = ""

    for call_num in range(_MAX_AGENT_CALLS):
        result = await llm.complete(messages, tools=_GENERATOR_TOOLS, temperature=0.3)
        tool_calls = result.get("tool_calls")

        if not tool_calls:
            final_content = (result["content"] or "").strip()
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
        logger.info("generate/chat call=%d tools=%s", call_num + 1, tool_names)

        for tc in tool_calls:
            try:
                inputs: dict = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                inputs = {}

            if tc["name"] == "propose_extraction_schema":
                tool_output = await _run_propose_schema(llm, messages)
            elif tc["name"] == "propose_app_config":
                tool_output, config_yaml = await _run_propose_config(llm, messages)
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
        logger.warning("generate/chat reached max_calls=%d without final answer", _MAX_AGENT_CALLS)
        final_content = result.get("content") or ""  # type: ignore[possibly-undefined]

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
