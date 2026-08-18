"""The account's company profile — collect it, read, edit, and delete.

  GET/PUT/DELETE /profile              the profile document itself
  POST /profile/interview/chat         one onboarding-interview turn
  POST /profile/interview/chat/stream  the same turn, streamed as SSE

The company profile is stable org-wide context a customer supplies once (who they
are, jurisdictions, regulators, risk appetite, house style). It is *account*
scoped, not namespace scoped, so these routes carry no ``{namespace}`` segment —
like ``GET /applications``, they address the whole account.

Two stores back one resource, following the ``skill_records`` precedent: the
markdown body lives in the system document store (``AccountProfileStore``), the
edit metadata in the ``profile_records`` table. A write touches both, then pushes
the new text into the account's already-built app instances — see
:func:`apply_profile_to_live_apps`.

The interview lives here, next to the storage it writes through, rather than in
the app-generator chat: a freshly-minted account is already provisioned with a
namespace and an app (``api/provisioning.py``), so the user who most needs
onboarding may never open the Build tab. Its script — the questions and the
profile template both — is a skill resolved by name; ``cogbase/core/onboarding.py``
supplies only the frame around it. This module is the only LLM-driven writer of
the profile — the generator neither writes nor reads one.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import (
    AccountIdDep,
    AccountProfileStoreDep,
    AppCacheDep,
    InterviewSkillResolverDep,
    SkillRegistryDep,
    SystemResourcesDep,
    SystemStoreDep,
    principal_claims,
)
from api.app_cache import AppCache
from api.models import (
    CompanyProfileResponse,
    InterviewChatRequest,
    InterviewChatResponse,
    UpdateCompanyProfileRequest,
)
from api.system_resources import SystemResources
from api.system_store import ProfileRecord, SystemStore
from cogbase.core.onboarding import (
    INTERVIEW_TOOLS,
    SAVE_COMPANY_PROFILE_TOOL_NAME,
    InterviewSkillResolver,
    build_interview_system_prompt,
    resolve_interview_script,
    resolve_interview_skill_name,
)
from cogbase.core.profile import AccountProfileStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["profile"])

#: The interview is a conversation with one tool, so a turn should take one or two
#: LLM rounds. The ceiling only exists to stop a model that loops on a failing save.
_MAX_INTERVIEW_CALLS = 6


def apply_profile_to_live_apps(
    app_cache: AppCache, account_id: str, markdown: str | None
) -> int:
    """Push an edited profile into the account's cached app instances.

    The profile is read once per app build (``api/factory.py``) and held on the
    query runner for the life of the instance, so an edit has to reach the live
    instances somehow. Hot-patching beats evicting them: the profile is a page of
    framing prose, while a cached app owns warm memory tiers, a skill registry,
    and wired workflows that a rebuild would pay for again on the next query.
    ``markdown=None`` clears the block (the delete path).

    Node-local, like the cache itself. Other nodes keep serving the previous
    profile until their app-cache TTL expires — 60s (``api/app_cache.py``). For a
    rarely-edited page of prose that window is accepted rather than engineered
    away with cross-node invalidation.

    Returns the number of instances patched (for logging).
    """
    apps = app_cache.apps_for_account(account_id)
    for app in apps:
        app.set_account_profile(markdown)
    return len(apps)


# ---------------------------------------------------------------------------
# Conversation-safe access: best-effort read, tool-shaped write
# ---------------------------------------------------------------------------


def profile_store_or_none(
    system_resources: SystemResources,
) -> AccountProfileStore | None:
    """The profile store, or ``None`` in a deployment with no document store.

    The deliberate opposite of :data:`AccountProfileStoreDep`, which 503s. A
    conversation must survive a deployment that cannot hold profiles; only the
    explicit ``PUT /profile`` should fail loudly, because a write that silently
    does nothing is worse than an error.
    """
    store = system_resources.document_store
    return AccountProfileStore(store) if store is not None else None


async def load_profile_best_effort(
    profile_store: AccountProfileStore | None, account_id: str, log_prefix: str
) -> str | None:
    """Read the profile, treating any failure as "no profile yet".

    A store hiccup should cost the account a re-ask, not a dead turn.
    """
    if profile_store is None:
        return None
    try:
        return await profile_store.load(account_id)
    except Exception:
        logger.warning(
            "%s account profile load failed account=%s", log_prefix, account_id,
            exc_info=True,
        )
        return None


async def save_profile_from_tool(
    args: dict,
    *,
    account_id: str,
    profile_store: AccountProfileStore | None,
    system_store: SystemStore | None,
    app_cache: AppCache | None,
    log_prefix: str,
) -> tuple[str, str | None]:
    """Persist a ``save_company_profile`` tool call. Returns ``(tool_result, saved)``.

    Every failure is reported back *as the tool result* rather than raised: the
    model is mid-conversation, and the useful behaviour is to tell the user what
    happened and carry on, not to kill the turn. ``saved`` is the persisted
    markdown, or ``None`` when nothing was written.

    The success path also hot-patches the account's live apps, which is what makes
    the auto-provisioned ``contract-analyst`` pick up a profile written seconds
    ago without a rebuild.
    """
    markdown = (args.get("markdown") or "").strip()
    if not markdown:
        return "No profile text was provided — nothing was saved. Ask again, then retry.", None
    if profile_store is None:
        return (
            "Company profiles are not available on this deployment (no document "
            "store configured). Nothing was saved — tell the user, and carry on.",
            None,
        )

    try:
        await profile_store.save(account_id, markdown)
    except ValueError as exc:  # over MAX_PROFILE_BYTES
        return f"Profile not saved: {exc}. Shorten it to the essentials and try again.", None
    except Exception:
        logger.exception("%s company profile save failed account=%s", log_prefix, account_id)
        return (
            "Saving the company profile failed. Tell the user it did not save, and "
            "carry on.",
            None,
        )

    if system_store is not None:
        await system_store.save_profile_record(account_id, source="interview")
    patched = apply_profile_to_live_apps(app_cache, account_id, markdown) if app_cache else 0
    logger.info(
        "%s saved company profile account=%s bytes=%d live_apps_patched=%d",
        log_prefix, account_id, len(markdown.encode("utf-8")), patched,
    )
    return (
        "Company profile saved — every app in this account reads it, and these "
        "questions will not be asked again. Summarize what you captured in a few "
        "lines and ask what you got wrong.",
        markdown,
    )


def _updated_by(authorization: str | None) -> str | None:
    """Identify the editing principal, when the request carries a valid token.

    Returns ``None`` in ``dev`` mode (header-only tenancy, no principal to name),
    which is why ``profile_records.updated_by`` is nullable.
    """
    claims = principal_claims(authorization)
    if claims is None:
        return None
    return claims.get("email") or claims.get("sub")


def _response(
    markdown: str | None, record: ProfileRecord | None
) -> CompanyProfileResponse:
    return CompanyProfileResponse(
        markdown=markdown,
        exists=markdown is not None,
        updated_at=record.updated_at if record else None,
        updated_by=record.updated_by if record else None,
        source=record.source if record else None,
    )


@router.get("", response_model=CompanyProfileResponse)
async def get_profile(
    account_id: AccountIdDep,
    profile_store: AccountProfileStoreDep,
    system_store: SystemStoreDep,
) -> CompanyProfileResponse:
    """Return the calling account's company profile.

    An account that has not been through onboarding gets ``200`` with
    ``exists: false`` and a null body — absence is a state the UI branches on,
    not an error.
    """
    markdown = await profile_store.load(account_id)
    record = await system_store.get_profile_record(account_id) if markdown else None
    return _response(markdown, record)


@router.put("", response_model=CompanyProfileResponse)
async def update_profile(
    account_id: AccountIdDep,
    profile_store: AccountProfileStoreDep,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    body: UpdateCompanyProfileRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> CompanyProfileResponse:
    """Replace the account's company profile with *markdown*.

    Oversized bodies are refused with 413: past ``MAX_PROFILE_BYTES`` the profile
    stops being framing context and starts crowding out the documents an answer
    is grounded in. The cap is enforced in the store, so the interview's writer
    is held to the same limit.
    """
    try:
        await profile_store.save(account_id, body.markdown)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        )

    record = await system_store.save_profile_record(
        account_id, source="manual", updated_by=_updated_by(authorization)
    )
    patched = apply_profile_to_live_apps(app_cache, account_id, body.markdown)
    logger.info(
        "updated company profile account=%s bytes=%d live_apps_patched=%d",
        account_id, len(body.markdown.encode("utf-8")), patched,
    )
    return _response(body.markdown, record)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_profile(
    account_id: AccountIdDep,
    profile_store: AccountProfileStoreDep,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
) -> None:
    """Remove the account's profile — body, index row, and the live prompt block.

    Idempotent: deleting an account that has no profile succeeds.
    """
    await profile_store.delete(account_id)
    await system_store.delete_profile_record(account_id)
    patched = apply_profile_to_live_apps(app_cache, account_id, None)
    logger.info(
        "deleted company profile account=%s live_apps_patched=%d", account_id, patched
    )


# ---------------------------------------------------------------------------
# The onboarding interview
# ---------------------------------------------------------------------------


async def _interview_turn_events(
    body: InterviewChatRequest,
    system_resources: SystemResources,
    *,
    account_id: str,
    log_prefix: str,
    system_store: SystemStore | None = None,
    app_cache: AppCache | None = None,
    skill_registry=None,
    interview_skill_resolver: InterviewSkillResolver | None = None,
):
    """Run one interview turn, yielding ``token`` / ``result`` / ``error`` events.

    Stateless, like the generator chat: the client owns the user/assistant
    history, and tool messages live only within this turn. An existing profile
    makes the turn a re-run over what is already saved rather than a second mode.
    """
    llm = system_resources.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM configured on the system")

    from cogbase.llms.base import ChatMessage as LLMChatMessage

    logger.info(
        "%s start account=%s history=%d", log_prefix, account_id, len(body.history)
    )

    # Which script, then the script. The deployment's resolver (if any) answers the
    # first question per account; without one every account in the process gets the
    # env-level default, which is right for a single-vertical deployment and wrong
    # for anything serving two.
    try:
        skill_name = await resolve_interview_skill_name(
            interview_skill_resolver, account_id
        )
    except Exception as exc:
        logger.exception("%s interview resolver failed account=%s", log_prefix, account_id)
        # Not a fallback to the default: a resolver that raised did not say "use the
        # default", and the default is one vertical's questionnaire saved as this
        # account's profile forever.
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not determine which onboarding interview applies to this "
                f"account: {exc}"
            ),
        ) from exc

    script = resolve_interview_script(skill_registry, account_id, name=skill_name)
    if script is None:
        # The interview *is* its script — questions and profile template both live
        # in a SKILL.md, so there is nothing to fall back to. Failing loudly beats
        # improvising an interview whose answers are then saved forever.
        raise HTTPException(
            status_code=503,
            detail=(
                f"No onboarding interview script is registered (expected a skill "
                f"named '{skill_name}')."
            ),
        )

    profile_store = profile_store_or_none(system_resources)
    existing = await load_profile_best_effort(profile_store, account_id, log_prefix)
    system_prompt = build_interview_system_prompt(script, existing_profile=existing)

    messages: list[LLMChatMessage] = (
        [{"role": "system", "content": system_prompt}]
        + [{"role": m.role, "content": m.content} for m in body.history]
        + [{"role": "user", "content": body.text}]
    )

    saved_markdown: str | None = None
    final_content = ""
    streamed_chunks: list[str] = []

    try:
        for call_num in range(_MAX_INTERVIEW_CALLS):
            streamed_chunks = []
            result = None
            async for chunk in llm.complete_stream(
                messages, tools=INTERVIEW_TOOLS, temperature=0.3
            ):
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
            tool_message = {"role": "tool", "tool_call_id": tc["id"], "content": "Running..."}
            messages.append(tool_message)

            try:
                args = json.loads(tc.get("arguments") or "{}")
            except (json.JSONDecodeError, ValueError):
                args = {}

            if tc["name"] != SAVE_COMPANY_PROFILE_TOOL_NAME:
                logger.warning("%s unknown tool=%s", log_prefix, tc["name"])
                tool_message["content"] = (
                    f"Unknown tool '{tc['name']}'. The only tool available here is "
                    f"{SAVE_COMPANY_PROFILE_TOOL_NAME}."
                )
                continue

            tool_message["content"], saved = await save_profile_from_tool(
                args,
                account_id=account_id,
                profile_store=profile_store,
                system_store=system_store,
                app_cache=app_cache,
                log_prefix=log_prefix,
            )
            saved_markdown = saved or saved_markdown
        else:
            final_content = "".join(streamed_chunks).strip()
            logger.warning(
                "%s reached max_calls=%d without a final answer",
                log_prefix, _MAX_INTERVIEW_CALLS,
            )

        logger.info(
            "%s turn=%d profile_saved=%s content=%d",
            log_prefix, len(body.history) + 1, saved_markdown is not None, len(final_content),
        )
        yield {
            "type": "result",
            "result": {"content": final_content, "markdown": saved_markdown},
        }
    except Exception as exc:
        logger.exception("%s failed", log_prefix)
        exc_module = type(exc).__module__ or ""
        detail = (
            "LLM unavailable"
            if exc_module.startswith("openai") or exc_module.startswith("httpx")
            else "interview turn failed"
        )
        yield {"type": "error", "error": detail}


@router.post("/interview/chat", response_model=InterviewChatResponse)
async def interview_chat(
    account_id: AccountIdDep,
    body: InterviewChatRequest,
    system_resources: SystemResourcesDep,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    skill_registry: SkillRegistryDep,
    interview_skill_resolver: InterviewSkillResolverDep = None,
) -> InterviewChatResponse:
    """One onboarding-interview turn.

    Account-scoped and stateless — the client holds the message history, the
    server runs the agent loop. When the model has enough it calls
    ``save_company_profile``; the response's ``profile_saved`` tells the UI to
    dismiss the onboarding card.
    """
    content = ""
    markdown: str | None = None
    async for event in _interview_turn_events(
        body,
        system_resources,
        account_id=account_id,
        log_prefix="profile/interview",
        system_store=system_store,
        app_cache=app_cache,
        skill_registry=skill_registry,
        interview_skill_resolver=interview_skill_resolver,
    ):
        if event["type"] == "result":
            content = event["result"]["content"]
            markdown = event["result"]["markdown"]
        elif event["type"] == "error":
            raise HTTPException(status_code=502, detail=event["error"])
    return InterviewChatResponse(
        content=content, profile_saved=markdown is not None, markdown=markdown
    )


@router.post("/interview/chat/stream")
async def interview_chat_stream(
    account_id: AccountIdDep,
    body: InterviewChatRequest,
    system_resources: SystemResourcesDep,
    system_store: SystemStoreDep,
    app_cache: AppCacheDep,
    skill_registry: SkillRegistryDep,
    interview_skill_resolver: InterviewSkillResolverDep = None,
) -> StreamingResponse:
    """Stream an interview turn as Server-Sent Events.

    Emits ``{"token": ...}`` as the model talks, then a final
    ``{"done": true, "content": ..., "profile_saved": ..., "markdown": ...}``.
    """
    async def event_stream():
        try:
            async for event in _interview_turn_events(
                body,
                system_resources,
                account_id=account_id,
                log_prefix="profile/interview/stream",
                system_store=system_store,
                app_cache=app_cache,
                skill_registry=skill_registry,
                interview_skill_resolver=interview_skill_resolver,
            ):
                if event["type"] == "token":
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps({'error': event['error']})}\n\n"
                else:
                    markdown = event["result"]["markdown"]
                    yield "data: " + json.dumps({
                        "done": True,
                        "content": event["result"]["content"],
                        "profile_saved": markdown is not None,
                        "markdown": markdown,
                    }) + "\n\n"
        except HTTPException as exc:
            yield f"data: {json.dumps({'error': exc.detail})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
