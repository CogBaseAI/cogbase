"""App generator endpoints — agentic, conversational config.yaml creation.

The LLM drives the conversation: it asks what it needs, proposes config sections
as they become clear, and refines on feedback. No explicit phases or session state
on the server — the client owns the full message history.

Endpoints
---------
  POST /generate/chat    stateless chat turn; LLM may embed a ---CONFIG--- block
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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])

# ---------------------------------------------------------------------------
# Block markers — stripped before display; stored verbatim in history for LLM context.
# ---------------------------------------------------------------------------

_CONFIG_START = "---CONFIG---"
_CONFIG_END = "---END CONFIG---"
_SCHEMA_START = "---EXTRACTION SCHEMA---"
_SCHEMA_END = "---END EXTRACTION SCHEMA---"
_RESOLVED_START = "---SCHEMA RESOLVED---"
_RESOLVED_END = "---END SCHEMA RESOLVED---"

_MAX_SCHEMA_RETRIES = 3

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an agentic CogBase application generator. Help the user build a complete, \
correct CogBase app configuration through natural conversation. You drive the process.

CogBase applications ingest documents, extract structured facts with an LLM, and answer \
natural-language questions via semantic search and structured lookup.

## How to work

1. Ask targeted questions — no more than 2-3 per turn — to understand:
   - What documents will be ingested (type, format, content)
   - What structured fields to extract (name, data type, description for each field)
   - What kinds of queries users will run:
       * Exact/filtered lookup over extracted fields → needs extract-structured step
       * Semantic search over document text → needs chunk-embed-upsert step
       * High-level summary or topic queries → needs document-embed-upsert step

2. Propose and finalize the extraction schema(s) before writing the full config.
   Use full JSON Schema (YAML-encoded) for every collection. One block covers all:

---EXTRACTION SCHEMA---
<collection_name>:
  type: object
  properties:
    <scalar_field>:
      anyOf:
        - type: <string|integer|number|boolean>
        - type: "null"
      description: "<what this field captures>"
    <nested_object_field>:
      anyOf:
        - type: object
          properties:
            <subfield>:
              anyOf: [{type: string}, {type: "null"}]
              description: "..."
        - type: "null"
      description: "<what the nested object captures>"
    <list_of_objects_field>:
      type: array
      items:
        type: object
        properties:
          <subfield>:
            type: string
            description: "..."
      description: "<what each list item represents>"
      default: []
<another_collection>:
  type: object
  properties:
    ...
---END EXTRACTION SCHEMA---

   Rules:
   - Top-level keys are structured collection names (matching the config exactly)
   - Each collection value is a JSON Schema object (must have type: object)
   - Do not include doc_id — it is injected automatically
   - Nullable/optional scalars use anyOf with null
   - List fields use type: array; add default: [] so they are never null
   - Nested objects use type: object with inline properties (or $defs for reuse)

   The system validates immediately by building a Pydantic model from each schema.
   Errors will be returned; fix and re-propose the full ---EXTRACTION SCHEMA--- block.
   Do not propose the full config until confirmed.

   After validation, the system appends a ---SCHEMA RESOLVED--- block to this message
   with the exact JSON strings to paste into the config. Use them verbatim.

3. Propose the full config once the schema is confirmed (---SCHEMA RESOLVED--- block
   has appeared). Use the JSON strings from that block for extraction_schema and schema.

4. When proposing or updating a config, embed it in your response like this:

---CONFIG---
name: my-app
... full config.yaml ...
---END CONFIG---

   Always include a plain-language explanation alongside: what you set up and why,
   and what (if anything) you still need.

## Config rules

1. name must be kebab-case (lowercase, alphanumeric, hyphens only)
2. chunk-embed-upsert is always the first pipeline step
3. Do NOT include doc_id in extraction schemas — it is injected automatically
4. All content is INLINE — do not use .json or .txt filenames as values anywhere
5. Pipeline step collections must exactly match declared vector/structured collection names
6. Use snake_case for all collection names and field names
7. Copy extraction_schema and schema values verbatim from ---SCHEMA RESOLVED---

## Config format

name: <kebab-case-name>

vector_collections:
  - name: <snake_case>
    description: "<shown to the LLM as context during retrieval>"

structured_collections:
  - name: <snake_case>
    description: "<shown to the LLM as context during lookup>"
    schema: '<record_schema JSON string from ---SCHEMA RESOLVED--->'
    primary_fields: [doc_id]

pipelines:
  - name: <name>
    steps:
      - tool: chunk-embed-upsert
        collection: <vector_collection>
        chunker:
          type: langchain

      - tool: extract-structured          # include only if structured extraction is needed
        collection: <structured_collection>
        extractor:
          type: llm
          extraction_schema: '<extraction_schema JSON string from ---SCHEMA RESOLVED--->'
          prompt: |
            <System instructions for the extraction LLM. Be specific.>

      - tool: document-embed-upsert       # include only if summary/topic queries are needed
        collection: <vector_collection>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_block(text: str, start_marker: str, end_marker: str) -> tuple[str, str | None]:
    """Extract a delimited block from *text*, returning (text_without_block, block | None)."""
    if start_marker not in text:
        return text, None

    before, rest = text.split(start_marker, 1)
    if end_marker in rest:
        raw_block, after = rest.split(end_marker, 1)
    else:
        raw_block, after = rest, ""

    content = _strip_fences(raw_block)
    display = (before.strip() + ("\n\n" + after.strip() if after.strip() else "")).strip()
    return display, content


def _extract_config(text: str) -> tuple[str, str | None]:
    """Split LLM response into (display_text, config_yaml | None)."""
    return _extract_block(text, _CONFIG_START, _CONFIG_END)


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


def _validate_extraction_schema(schema_dict: dict, collection_name: str) -> list[str]:
    """Validate a JSON Schema dict for use as an extraction schema.

    Builds an actual Pydantic model from it so errors are structural, not syntactic.
    """
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
    """Parse a YAML-encoded schema block (collection_name → JSON Schema) and validate all collections.

    Returns (schemas_dict, errors). On parse failure schemas_dict is None.
    """
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, [f"Schema block is not valid YAML: {exc}"]
    if not isinstance(parsed, dict):
        return None, ["Schema block must be a mapping of collection_name → JSON Schema object"]
    errors: list[str] = []
    for collection_name, schema_dict in parsed.items():
        errors.extend(_validate_extraction_schema(schema_dict, collection_name))
    return parsed, errors


def _build_resolved_block(schemas: dict) -> str:
    """Produce a ---SCHEMA RESOLVED--- block with verbatim JSON strings for the config."""
    lines = [_RESOLVED_START]
    for name, schema_dict in schemas.items():
        ext_json = json.dumps(schema_dict, separators=(",", ":"))
        rec_json = json.dumps(_make_record_schema(schema_dict), separators=(",", ":"))
        lines.append(f"{name}:")
        lines.append(f"  extraction_schema: '{ext_json}'")
        lines.append(f"  schema: '{rec_json}'")
    lines.append(_RESOLVED_END)
    return "\n".join(lines)


def _serialize_config(config: AppConfig) -> str:
    return yaml.dump(
        config.model_dump(by_alias=True, mode="json"),
        allow_unicode=True,
        default_flow_style=False,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=GenerateChatResponse)
async def chat(
    body: GenerateChatRequest,
    system_resources: SystemResourcesDep,
) -> GenerateChatResponse:
    """One turn of the generator conversation.

    The client maintains the full message history and sends it each call.
    The LLM may embed a ``---SCHEMA---`` block (validated here, with auto-retry on
    errors) and/or a ``---CONFIG---`` block.  The client should store the full
    ``content`` (markers included) in its local history so the LLM retains context.
    """
    llm = system_resources.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM configured on the system")

    from cogbase.llms.base import ChatMessage as LLMChatMessage
    messages: list[LLMChatMessage] = [
        {"role": m.role, "content": m.content} for m in body.history
    ] + [{"role": "user", "content": body.text}]

    validated_schema: dict | None = None
    full_content: str = ""

    for attempt in range(_MAX_SCHEMA_RETRIES + 1):
        result = await llm.complete(
            [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
            temperature=0.3,
        )
        full_content = (result["content"] or "").strip()

        _, schema_raw = _extract_block(full_content, _SCHEMA_START, _SCHEMA_END)
        if schema_raw is None:
            break  # no schema block proposed — nothing to validate

        schemas, errors = _parse_and_validate_schemas(schema_raw)
        if not errors:
            validated_schema = schemas
            logger.info(
                "generate/chat schemas validated collections=%s",
                list(schemas),
            )
            break

        if attempt < _MAX_SCHEMA_RETRIES:
            error_text = "Schema validation errors:\n" + "\n".join(f"- {e}" for e in errors)
            error_text += "\nPlease fix the schema and re-propose the full ---EXTRACTION SCHEMA--- block."
            logger.info("generate/chat schema attempt=%d errors=%d", attempt + 1, len(errors))
            messages = messages + [
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": error_text},
            ]
        else:
            logger.warning("generate/chat schema still invalid after %d retries", _MAX_SCHEMA_RETRIES)

    if validated_schema is not None:
        full_content = full_content + "\n\n" + _build_resolved_block(validated_schema)

    display_text, _ = _extract_block(full_content, _SCHEMA_START, _SCHEMA_END)
    display_text, _ = _extract_block(display_text, _RESOLVED_START, _RESOLVED_END)
    display_text, config_yaml = _extract_config(display_text)

    logger.info(
        "generate/chat turn=%d schema_confirmed=%s config_proposed=%s",
        len(body.history) + 1,
        validated_schema is not None,
        config_yaml is not None,
    )
    return GenerateChatResponse(
        content=full_content,
        config_yaml=config_yaml,
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
