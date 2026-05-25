"""App generator endpoints.

  POST /generate/chat        stateless chat turn; agent loop runs server-side
  POST /generate/chat/stream streaming chat turn (SSE)
  POST /generate/deploy      create and activate an application from a config_yaml
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

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
from cogbase.core.app_generator import (
    GENERATOR_TOOLS,
    SYSTEM_PROMPT,
    propose_app_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generate"])

_MAX_AGENT_CALLS = 10

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _categorize_error(exc: Exception) -> str:
    exc_module = type(exc).__module__ or ""
    if exc_module.startswith("openai") or exc_module.startswith("httpx"):
        return "LLM unavailable"
    return "stream failed"


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
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + [{"role": m.role, "content": m.content} for m in body.history]
        + [{"role": "user", "content": body.text}]
    )

    validated_config_yaml: str | None = None
    final_content: str = ""
    streamed_chunks: list[str] = []

    try:
        for call_num in range(_MAX_AGENT_CALLS):
            streamed_chunks = []
            result = None
            async for chunk in llm.complete_stream(messages, tools=GENERATOR_TOOLS, temperature=0.3):
                if isinstance(chunk, str):
                    streamed_chunks.append(chunk)
                    yield {"type": "token", "token": chunk}
                else:
                    result = chunk

            tool_calls = result.get("tool_calls") if result else None

            if not tool_calls:
                final_content = "".join(streamed_chunks).strip()
                break

            tc = tool_calls[0]
            logger.info("%s call=%d tool=%s", log_prefix, call_num + 1, tc["name"])

            messages.append({
                "role": "assistant",
                "content": result.get("content"),
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }],
            })

            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Running..."})

            try:
                args = json.loads(tc.get("arguments") or "{}")
            except (json.JSONDecodeError, ValueError):
                args = {}
            needs_workflow = bool(args.get("needs_workflow", False))

            generation_context = ""
            async for event in propose_app_config(llm, messages, needs_workflow=needs_workflow):
                if event["type"] == "token":
                    yield {"type": "token", "token": event["token"]}
                else:
                    generation_context = event["generation_context"]
                    validated_config_yaml = event["config_yaml"]

            # Final LLM turn: user-facing summary or error explanation (no tools)
            messages.append({"role": "user", "content": generation_context})
            streamed_chunks = []
            async for chunk in llm.complete_stream(messages, tools=[], temperature=0.3):
                if isinstance(chunk, str):
                    streamed_chunks.append(chunk)
                    yield {"type": "token", "token": chunk}
            final_content = "".join(streamed_chunks).strip()
            break
        else:
            final_content = "".join(streamed_chunks).strip()
            logger.warning(
                "%s reached max_calls=%d without final answer, final_content=%s, messages=%s",
                log_prefix,
                _MAX_AGENT_CALLS,
                final_content,
                messages,
            )

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
    except Exception as exc:
        logger.exception("%s failed", log_prefix)
        yield {"type": "error", "error": _categorize_error(exc)}


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

    stored_yaml = config.to_yaml()
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
        app = await build_app(config, system=system_resources, app_status=record.status, task_store=system_store)
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
