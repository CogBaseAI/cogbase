"""App generator endpoints — conversational config.yaml creation via LLM."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml
from fastapi import APIRouter, HTTPException, status

from api.dependencies import AppCacheDep, SystemResourcesDep, SystemStoreDep
from api.factory import build_app
from api.models import (
    ConfigSummary,
    DeployResponse,
    GenerateRequest,
    GenerateResponse,
    ReviseRequest,
    ReviseResponse,
    StructuredCollectionSummary,
)
from api.system_store import AppRecord
from cogbase.config.config import AppConfig
from cogbase.llms.base import ChatMessage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    session_id: str
    description: str
    config_yaml: str
    messages: list[ChatMessage] = field(default_factory=list)


_sessions: dict[str, _Session] = {}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a CogBase application configuration generator.

CogBase builds AI applications that ingest documents, extract structured facts, \
and answer natural-language questions. You produce complete, valid config.yaml files \
from a plain-language description.

## Config structure

name: <kebab-case app name>

vector_collections:
  - name: <snake_case_name>
    description: "<what this index is for — shown to the LLM during retrieval>"

structured_collections:
  - name: <snake_case_name>
    description: "<what this collection stores — shown to the LLM>"
    schema: |
      {
        "type": "object",
        "required": ["doc_id"],
        "properties": {
          "doc_id": {"type": "string"},
          "<field>": {"type": ["<type>", "null"], "description": "<description>"}
        }
      }
    primary_fields: [doc_id]

pipelines:
  - name: <name>
    steps:
      - tool: chunk-embed-upsert
        collection: <vector_collection_name>
        chunker:
          type: langchain

      - tool: extract-structured        # omit if no structured extraction needed
        collection: <structured_collection_name>
        extractor:
          type: llm
          extraction_schema: |
            {
              "type": "object",
              "properties": {
                "<field>": {"type": ["<type>", "null"], "description": "<description>"}
              }
            }
          prompt: |
            <System instructions for the extraction LLM. Be specific about what to extract.>

      - tool: document-embed-upsert     # omit if no summary or topic-level queries needed
        collection: <vector_collection_name>

## Critical rules

1. App name must be lowercase alphanumeric + hyphens only (kebab-case)
2. chunk-embed-upsert is always the first pipeline step
3. schema (record schema) MUST include "doc_id" as a required string field
4. extraction_schema MUST NOT include "doc_id" — the extractor injects it automatically
5. Every extractable field must be nullable: use ["<type>", "null"] not just "<type>"
6. All content is INLINE — do not use .json or .txt filenames as values
7. Pipeline step collections must match declared vector/structured collection names
8. Use snake_case for collection names and field names
9. Include informative descriptions on every extraction_schema field

## Output

Output ONLY the raw config.yaml. No markdown code fences. No explanation. No trailing text.
The first line must be: name: <app-name>
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


def _summarize(config_yaml: str) -> ConfigSummary:
    try:
        data = yaml.safe_load(config_yaml) or {}
    except Exception:
        data = {}

    name = data.get("name", "unknown")

    vc_names = [
        vc["name"] for vc in data.get("vector_collections", []) if isinstance(vc, dict)
    ]

    sc_summaries: list[StructuredCollectionSummary] = []
    for sc in data.get("structured_collections", []):
        if not isinstance(sc, dict):
            continue
        sc_name = sc.get("name", "?")
        schema_raw = sc.get("schema") or ""
        try:
            schema = json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
            fields = list((schema.get("properties") or {}).keys())
        except Exception:
            fields = []
        sc_summaries.append(StructuredCollectionSummary(name=sc_name, fields=fields))

    step_tools: list[str] = []
    seen: set[str] = set()
    for pipeline in data.get("pipelines", []):
        for step in pipeline.get("steps") or []:
            tool = step.get("tool", "")
            if tool and tool not in seen:
                step_tools.append(tool)
                seen.add(tool)

    return ConfigSummary(
        name=name,
        vector_collections=vc_names,
        structured_collections=sc_summaries,
        pipeline_steps=step_tools,
    )


def _diff(before: ConfigSummary, after: ConfigSummary) -> list[str]:
    changes: list[str] = []

    if before.name != after.name:
        changes.append(f"renamed: {before.name!r} → {after.name!r}")

    before_vc = set(before.vector_collections)
    after_vc = set(after.vector_collections)
    for vc in sorted(after_vc - before_vc):
        changes.append(f"+ vector collection: {vc}")
    for vc in sorted(before_vc - after_vc):
        changes.append(f"- vector collection: {vc}")

    before_sc = {sc.name: sc for sc in before.structured_collections}
    after_sc = {sc.name: sc for sc in after.structured_collections}
    for n in sorted(set(after_sc) - set(before_sc)):
        changes.append(f"+ structured collection: {n}")
    for n in sorted(set(before_sc) - set(after_sc)):
        changes.append(f"- structured collection: {n}")
    for n in sorted(set(before_sc) & set(after_sc)):
        b_fields = set(before_sc[n].fields)
        a_fields = set(after_sc[n].fields)
        for f in sorted(a_fields - b_fields):
            changes.append(f"+ field in {n}: {f}")
        for f in sorted(b_fields - a_fields):
            changes.append(f"- field in {n}: {f}")

    before_steps = set(before.pipeline_steps)
    after_steps = set(after.pipeline_steps)
    for s in sorted(after_steps - before_steps):
        changes.append(f"+ pipeline step: {s}")
    for s in sorted(before_steps - after_steps):
        changes.append(f"- pipeline step: {s}")

    return changes or ["no structural changes"]


def _serialize_config(config: AppConfig) -> str:
    return yaml.dump(
        config.model_dump(by_alias=True, mode="json"),
        allow_unicode=True,
        default_flow_style=False,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=GenerateResponse, status_code=status.HTTP_201_CREATED)
async def generate(
    body: GenerateRequest,
    system_resources: SystemResourcesDep,
) -> GenerateResponse:
    """Start a generator session from a natural-language description.

    The LLM produces a complete draft config.yaml. Refine it with
    POST /generate/{session_id}/revise, then deploy via
    POST /generate/{session_id}/deploy.
    """
    llm = system_resources.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM configured on the system")

    messages: list[ChatMessage] = [{"role": "user", "content": body.description}]
    result = await llm.complete(
        [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
        temperature=0.2,
    )
    config_yaml = _strip_fences(result["content"] or "")
    messages.append({"role": "assistant", "content": config_yaml})

    session_id = str(uuid.uuid4())
    _sessions[session_id] = _Session(
        session_id=session_id,
        description=body.description,
        config_yaml=config_yaml,
        messages=messages,
    )

    summary = _summarize(config_yaml)
    logger.info("generator session created session_id=%s app=%s", session_id, summary.name)
    return GenerateResponse(session_id=session_id, config_yaml=config_yaml, summary=summary)


@router.post("/{session_id}/revise", response_model=ReviseResponse)
async def revise(
    session_id: str,
    body: ReviseRequest,
    system_resources: SystemResourcesDep,
) -> ReviseResponse:
    """Revise the generated config with natural-language feedback."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Generator session '{session_id}' not found")

    llm = system_resources.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM configured on the system")

    before = _summarize(session.config_yaml)

    session.messages.append({"role": "user", "content": body.feedback})
    result = await llm.complete(
        [{"role": "system", "content": _SYSTEM_PROMPT}] + session.messages,
        temperature=0.2,
    )
    updated_yaml = _strip_fences(result["content"] or "")
    session.messages.append({"role": "assistant", "content": updated_yaml})
    session.config_yaml = updated_yaml

    after = _summarize(updated_yaml)
    changes = _diff(before, after)
    logger.info("generator session revised session_id=%s changes=%s", session_id, changes)
    return ReviseResponse(config_yaml=updated_yaml, summary=after, changes=changes)


@router.post("/{session_id}/deploy", response_model=DeployResponse, status_code=status.HTTP_201_CREATED)
async def deploy(
    session_id: str,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    system_resources: SystemResourcesDep,
) -> DeployResponse:
    """Deploy the current config as a new CogBase application."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Generator session '{session_id}' not found")

    try:
        config = AppConfig.from_yaml(session.config_yaml)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Generated config is invalid: {exc}") from exc

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
        logger.info("deployed app name=%s from session_id=%s", config.name, session_id)
    except Exception as exc:
        logger.exception("deploy failed session_id=%s", session_id)
        record = record.model_copy(
            update={"status": "error", "error": str(exc), "updated_at": _now()}
        )

    await system_store.save_app(record)
    del _sessions[session_id]
    return DeployResponse(name=record.name, status=record.status, error=record.error)
