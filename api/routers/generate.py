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
# Config block markers
# The LLM embeds a config proposal in its response using these sentinels.
# The backend extracts it; the CLI never shows the raw markers.
# ---------------------------------------------------------------------------

_CONFIG_START = "---CONFIG---"
_CONFIG_END = "---END CONFIG---"

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

2. Propose the config as soon as you have enough to work with. Don't wait for every
   detail — propose early and refine. Whenever something changes, re-propose the full
   updated config.

3. When proposing or updating a config, embed it in your response like this:

---CONFIG---
name: my-app
... full config.yaml ...
---END CONFIG---

   The markers are stripped before display. Always include a plain-language explanation
   alongside the config: what you set up and why, and what (if anything) you still need.

## Config rules

1. name must be kebab-case (lowercase, alphanumeric, hyphens only)
2. chunk-embed-upsert is always the first pipeline step
3. schema (record schema) MUST include "doc_id" as a required string field
4. extraction_schema MUST NOT include "doc_id" — the extractor injects it automatically
5. Every extractable field must be nullable: use ["<type>", "null"], never just "<type>"
6. All content is INLINE — do not use .json or .txt filenames as values anywhere
7. Pipeline step collections must exactly match declared vector/structured collection names
8. Use snake_case for all collection names and field names

## Config format

name: <kebab-case-name>

vector_collections:
  - name: <snake_case>
    description: "<shown to the LLM as context during retrieval>"

structured_collections:
  - name: <snake_case>
    description: "<shown to the LLM as context during lookup>"
    schema: |
      {
        "type": "object",
        "required": ["doc_id"],
        "properties": {
          "doc_id": {"type": "string"},
          "<field>": {"type": ["<type>", "null"], "description": "<what this field captures>"}
        }
      }
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
          extraction_schema: |            # must NOT include doc_id
            {
              "type": "object",
              "properties": {
                "<field>": {"type": ["<type>", "null"], "description": "<desc>"}
              }
            }
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


def _extract_config(text: str) -> tuple[str, str | None]:
    """Split LLM response into (display_text, config_yaml | None).

    The full response (with markers) should be stored in conversation history so
    the LLM retains context of its previous proposals. Only the display_text is
    shown to the user.
    """
    if _CONFIG_START not in text:
        return text, None

    before, rest = text.split(_CONFIG_START, 1)
    if _CONFIG_END in rest:
        raw_config, after = rest.split(_CONFIG_END, 1)
    else:
        raw_config, after = rest, ""

    config_yaml = _strip_fences(raw_config)
    display = (before.strip() + ("\n\n" + after.strip() if after.strip() else "")).strip()
    return display, config_yaml


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
    The LLM may embed a ``---CONFIG---`` block in its response; if so,
    ``config_yaml`` is populated in the response and the client should store
    the full ``content`` (markers included) in its local history so the LLM
    retains context of its previous proposals.
    """
    llm = system_resources.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM configured on the system")

    from cogbase.llms.base import ChatMessage as LLMChatMessage
    messages: list[LLMChatMessage] = [
        {"role": m.role, "content": m.content} for m in body.history
    ] + [{"role": "user", "content": body.text}]

    result = await llm.complete(
        [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
        temperature=0.3,
    )
    full_content = (result["content"] or "").strip()
    display_text, config_yaml = _extract_config(full_content)

    logger.info(
        "generate/chat turn=%d config_proposed=%s",
        len(body.history) + 1,
        config_yaml is not None,
    )
    return GenerateChatResponse(content=full_content, config_yaml=config_yaml)


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
