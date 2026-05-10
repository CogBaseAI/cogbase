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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])

# ---------------------------------------------------------------------------
# Block markers — stripped before display; stored verbatim in history for LLM context.
# ---------------------------------------------------------------------------

_CONFIG_START = "---CONFIG---"
_CONFIG_END = "---END CONFIG---"
_SCHEMA_START = "---EXTRACTION SCHEMA---"
_SCHEMA_END = "---END EXTRACTION SCHEMA---"

_MAX_SCHEMA_RETRIES = 3

# Scalar types the generator supports.  Complex nested schemas are for
# manually authored configs and are not validated here.
_SCALAR_TYPES = {"string", "integer", "number", "boolean"}

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
   Group fields by structured collection name — one block covers all collections at once.
   If an app has multiple pipelines or a pipeline has multiple extract-structured steps,
   include every collection that needs extraction:

---EXTRACTION SCHEMA---
<collection_name>:
  <field_name>:
    type: ["<basetype>", "null"]
    description: "<what this field captures>"
<another_collection_name>:
  <field_name>:
    type: ["<basetype>", "null"]
    description: "<what this field captures>"
---END EXTRACTION SCHEMA---

   Rules:
   - Top-level keys must be structured collection names (matching the config exactly)
   - Supported base types: string, integer, number, boolean
   - Every field must be nullable: type is always ["<basetype>", "null"]
   - Every field must have a description
   - Do not include doc_id in any schema — it is injected automatically
   The system validates immediately. Errors will be returned; fix and re-propose the
   full ---EXTRACTION SCHEMA--- block. Do not propose the full config until confirmed.

3. Propose the full config once the schema is confirmed. Don't wait for every
   detail — propose early and refine. Whenever something changes, re-propose.

4. When proposing or updating a config, embed it in your response like this:

---CONFIG---
name: my-app
... full config.yaml ...
---END CONFIG---

   The markers are stripped before display. Always include a plain-language explanation
   alongside: what you set up and why, and what (if anything) you still need.

## Config rules

1. name must be kebab-case (lowercase, alphanumeric, hyphens only)
2. chunk-embed-upsert is always the first pipeline step
3. Do NOT include doc_id in extraction_schema — injected automatically; doc_id is
   added to the record automatically and does not need to appear anywhere in the config
4. Every extractable field must be nullable: use ["<type>", "null"], never just "<type>"
5. All content is INLINE — do not use .json or .txt filenames as values anywhere
6. Pipeline step collections must exactly match declared vector/structured collection names
7. Use snake_case for all collection names and field names

## Config format

name: <kebab-case-name>

vector_collections:
  - name: <snake_case>
    description: "<shown to the LLM as context during retrieval>"

structured_collections:
  - name: <snake_case>
    description: "<shown to the LLM as context during lookup>"
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
          extraction_schema: <collection_extraction_schema>
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


def _validate_schema_properties(properties: dict) -> list[str]:
    """Return a list of human-readable error strings for invalid schema fields."""
    errors: list[str] = []
    for name, field in properties.items():
        if name == "doc_id":
            errors.append("doc_id must not appear in the schema — it is injected automatically")
            continue
        t = field.get("type")
        if not isinstance(t, list):
            errors.append(
                f"'{name}': type must be a two-element list like [\"string\", \"null\"], got {t!r}"
            )
            continue
        if "null" not in t:
            errors.append(f"'{name}': type must include \"null\" (all fields must be nullable)")
        base_types = [x for x in t if x != "null"]
        if not base_types:
            errors.append(f"'{name}': type must have at least one non-null base type")
        unknown = [x for x in base_types if x not in _SCALAR_TYPES]
        if unknown:
            errors.append(
                f"'{name}': unsupported type(s) {unknown!r}. Supported: {sorted(_SCALAR_TYPES)}"
            )
        if not field.get("description", "").strip():
            errors.append(f"'{name}': description is required")
    return errors


def _parse_and_validate_schemas(raw: str) -> tuple[dict | None, list[str]]:
    """Parse a YAML schema block (collection_name → fields) and validate all collections.

    Returns (schemas_dict, errors) where schemas_dict maps collection name → properties.
    On parse failure, schemas_dict is None.
    """
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, [f"Schema is not valid YAML: {exc}"]
    if not isinstance(parsed, dict):
        return None, ["Schema must be a YAML mapping of collection_name → fields"]

    errors: list[str] = []
    for collection_name, fields in parsed.items():
        if not isinstance(fields, dict):
            errors.append(
                f"[{collection_name}] must be a mapping of field_name → {{type, description}}"
            )
            continue
        for err in _validate_schema_properties(fields):
            errors.append(f"[{collection_name}] {err}")
    return parsed, errors


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
            error_text += "\nPlease fix the schema and re-propose the full ---SCHEMA--- block."
            logger.info("generate/chat schema attempt=%d errors=%d", attempt + 1, len(errors))
            messages = messages + [
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": error_text},
            ]
        else:
            logger.warning("generate/chat schema still invalid after %d retries", _MAX_SCHEMA_RETRIES)

    display_text, _ = _extract_block(full_content, _SCHEMA_START, _SCHEMA_END)
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
