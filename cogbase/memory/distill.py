"""Distiller: the offline pipeline that promotes durable memory from logs.

Distillation is ``extract-structured`` pointed at session logs instead of
documents (see docs/long-term-memory.md): a session log is the "document" and
long-term records are the extracted "structured records".  It reuses the
extraction *pattern* (schema-validated JSON + retry) but not the
``Document`` / ``doc_id`` contract — output feeds :meth:`LongTermMemory.reconcile`,
not an upsert by primary key.

The flow:

1. :meth:`EpisodicMemory.replay` the whole session log (it is short and bounded).
2. Project the conversational thread (the shared
   :mod:`cogbase.memory.projection` helper) and resolve provenance.
3. LLM-extract candidate memories, each carrying the ``source_event_ids`` it was
   derived from and a snapshot of the deciding turns for self-containment.  When
   existing memories are front-loaded (the single-call default), the same call
   also decides each candidate's reconcile op against them.
4. Reconcile each candidate and return the affected ids:
   :meth:`LongTermMemory.reconcile_decided` applies the op the extractor already
   chose (one LLM call per session), or :meth:`LongTermMemory.reconcile` decides
   it per candidate (N+1 calls — auditable, opt-in via ``single_call=False``).

Runs offline / async (on session settle), never on the request path.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import jsonschema

from cogbase.llms.base import LLMBase
from cogbase.memory.episodic import EpisodicMemory
from cogbase.memory.long_term import LongTermMemory
from cogbase.memory.models import (
    EventRef,
    EventType,
    LongTermRecord,
    MemoryCandidate,
    MemoryEvent,
    MemoryKind,
    MemoryMessage,
    ReconcileOp,
    normalize_entities,
)
from cogbase.memory.projection import latest_distillation, project_thread

logger = logging.getLogger(__name__)

# Cap on a single cited event's serialized payload copied into a snapshot.
# A final_answer can cite large tool results (e.g. a structured query dump);
# copying them wholesale bloats the evidence_snapshot JSON column.  Oversized
# payloads are truncated to a self-contained summary instead.
_MAX_CITED_PAYLOAD_BYTES = 8192

# Per-kind confidence floor: a candidate the LLM scores below its kind's floor
# is abandoned before reconcile — too weak a belief to be worth embedding,
# recalling, and persisting.  A candidate whose score is missing or non-numeric
# is abandoned the same way (an unreliable extraction is not salvaged with a
# fabricated default).
#
# These floors sit above the kind's stakes-scaled minimum because for the
# auto-promoting kinds (preference, retrieval_hint) the auto-promote threshold is
# 0.0 (see long_term.AUTO_PROMOTE_CONFIDENCE) — so the floor *is* the de facto
# auto-active bar: anything that survives goes straight to ``active`` with no
# review.  A weak (e.g. 0.5) preference should be abandoned, not auto-promoted.
DEFAULT_MIN_CONFIDENCE: dict[MemoryKind, float] = {
    MemoryKind.CORRECTION: 0.7,
    MemoryKind.FACT: 0.6,
    MemoryKind.PREFERENCE: 0.7,
    MemoryKind.RETRIEVAL_HINT: 0.6,
}

# Floor applied to a kind missing from the table (defensive).
_FALLBACK_MIN_CONFIDENCE = 0.5

# Cap on deterministic auto-links added per candidate, so even a candidate that
# shares discriminative entities with many existing memories stays sparsely linked.
_MAX_AUTO_LINKS = 3

_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [k.value for k in MemoryKind],
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_seqs": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    # Masked integer ids referencing the `## Existing memories`
                    # block; resolved back to real memory_ids in _build_candidate.
                    "linked_memory_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    # The single-call reconcile decision: how this memory
                    # reconciles against the `## Existing memories` block.  Default
                    # ADD (omitted) promotes it as new; UPDATE/DELETE/NOOP name an
                    # existing memory via target_id (the same masked-id namespace
                    # as linked_memory_ids).  See _build_candidate.
                    "operation": {
                        "type": "string",
                        "enum": [op.value for op in ReconcileOp],
                    },
                    "target_id": {"type": ["integer", "null"]},
                    "revised_content": {"type": ["string", "null"]},
                    # Bounds are enforced by clamping in _parse_confidence rather
                    # than the schema: an out-of-range value should be clamped,
                    # not fail validation and drop the whole extraction batch.
                    "confidence": {"type": "number"},
                },
                "required": ["content", "kind", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}

def _strip_code_fence(content: str) -> str:
    """Drop a leading/trailing ```...``` fence the model adds despite the prompt."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    # Drop the opening fence line (``` or ```json) and a closing fence if present.
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _as_index(value: object) -> int | None:
    """Coerce a masked-id value to an int index, or ``None`` if it isn't one.

    The extractor's ``target_id`` / ``linked_memory_ids`` reference the masked
    integer ids of the `## Existing memories` block; a non-numeric or missing
    value resolves to no record (``None``) rather than raising.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_memories(parsed: object) -> list | None:
    """Normalize the model's output to the ``memories`` array, or ``None``.

    Mirrors the pipeline extractor's tolerance (see
    ``llm._try_unwrap_single_item_list``): ``model="mini"`` occasionally wraps
    the result one level too deep, e.g. ``{"memories": {"memories": [...]}}``,
    which is valid JSON but fails the array schema and drops the whole batch.
    Drill through such nested ``memories`` wrappers, and accept a bare top-level
    list, before validation.
    """
    # Drill through nested {"memories": {"memories": ...}} wrappers (bounded).
    for _ in range(5):
        if isinstance(parsed, dict) and "memories" in parsed:
            parsed = parsed["memories"]
        else:
            break
    return parsed if isinstance(parsed, list) else None


_PROMPT_INTRO_AND_KINDS = (
    "You distill durable, long-term memories from a conversation transcript "
    "between a user and an assistant.  Each transcript line is prefixed with its "
    "turn number, e.g. [3 user] or [4 assistant].\n\n"
    "Extract only knowledge worth remembering across future sessions:\n"
    "- preference: a stable interaction preference (e.g. 'prefers concise "
    "answers', 'always wants citations').\n"
    "- fact: a stable, reusable fact worth recalling in a future session — about "
    "the user, their project or organization, OR a subject-matter fact the USER "
    "asserted, supplied, or affirmed (not one the assistant merely retrieved and "
    "restated from the source documents).\n"
    "- correction: a fact the user explicitly corrected or confirmed.\n"
    "- retrieval_hint: a recurring intent-to-tool-chain or routing pattern worth "
    "recalling at query time.\n\n"
)

_PROMPT_RULES_AND_SCHEMA = (
    "Rules:\n"
    "- If an `## Existing memories` block appears below the transcript, it lists "
    "durable memories already captured from prior sessions, each tagged with an "
    "id (e.g. [id=0]). Use it to (a) reconcile each memory you extract against "
    "accumulated belief — set its `operation` (see below); and (b) link related "
    "memories — set `linked_memory_ids` (see below). Never extract a memory FROM "
    "the existing-memories block — every extraction must come from the transcript. "
    "A known entity does not mean every fact about it is captured: still extract a "
    "genuinely new fact about an entity that already appears in an existing "
    "memory, as a separate ADD.\n"
    "- Set `operation` on every memory to how it reconciles against the "
    "`## Existing memories` block. Choose exactly one:\n"
    "    - ADD (the default): the claim is genuinely new — no existing memory "
    "covers it. Use ADD whenever there is no existing-memories block or none "
    "applies; omit target_id.\n"
    "    - UPDATE: an existing memory already expresses this claim — reinforce it, "
    "or refine its wording. Set target_id to that memory's id; set "
    "revised_content only when the wording should actually change, otherwise omit "
    "it.\n"
    "    - DELETE: the claim CONTRADICTS an existing memory and should replace it. "
    "Set target_id to the contradicted memory's id (it will be superseded and "
    "this claim promoted in its place).\n"
    "    - NOOP: an existing memory already states this claim correctly and "
    "completely — nothing to change. Set target_id to that memory's id.\n"
    "  Prefer UPDATE over a duplicate ADD when an existing memory expresses the "
    "same claim; use DELETE only for a real contradiction, not a mere refinement. "
    "Set target_id only to an id shown in the `## Existing memories` block.\n"
    "- Set linked_memory_ids to the ids of existing memories a new memory is "
    "specifically related to: the same entity or topic, a follow-up or "
    "continuation event, an update to it, or a contradiction of it. Use only ids "
    "shown in the `## Existing memories` block; omit it (or use an empty array) "
    "when nothing is related. Do NOT link on a vague shared theme.\n"
    "- Do NOT extract trivial chit-chat or transient task mechanics.\n"
    "- A subject-matter fact is durable only when the USER is its source — they "
    "asserted it, corrected the assistant, or confirmed a claim. Do NOT distill a "
    "fact that originates solely in an assistant turn derived from document "
    "retrieval; the source documents already own that knowledge.\n"
    "- Write each memory as a single self-contained natural-language claim, "
    "understandable on its own: replace pronouns with the specific name or 'the "
    "user'.\n"
    "- Preserve every specific detail exactly as stated — proper nouns and titles "
    "(people, places, organizations, brands, products, books/films), exact numbers "
    "and quantities, and qualifiers. Never generalize a specific into a vague "
    "category: keep 'assistant manager' (not 'manager'), 'Ferrari 488 GTB' (not "
    "'sports car'), '416 pages' (not 'about 400'). The specific detail is usually "
    "what makes the memory findable later, so completeness beats brevity — never "
    "drop a proper noun or number to shorten a claim.\n"
    "- Capture transitions and changes, not just the end state: 'switched from X "
    "to Y', 'was promoted from X to Y', 'no longer does X'.\n"
    "- Preserve the exact meaning of what was said; do not reword it into a "
    "different claim ('didn't get to bed until 2am' means a late bedtime, not "
    "'slept until 2am'; 'used to love hiking' means they no longer do). "
    "Misreading the user is worse than not extracting at all.\n"
    "- Extract the content that was shared, not a description of the act of "
    "sharing it (e.g. 'the Bajimaya case contract was signed in 2015', not 'the "
    "user shared a case summary').\n"
    "- Resolve every relative time reference (e.g. 'yesterday', 'last week', "
    "'recently', 'two months ago') to an absolute date, computed against the "
    "observation date given above the transcript — NOT against today's date. "
    "Never store a vague or relative temporal reference; if a date cannot be "
    "resolved, drop the temporal qualifier rather than guessing.\n"
    "- Set entities to the named entities the memory is about (people, projects, "
    "organizations, systems), lowercase; empty array when there are none.\n"
    "- Set source_seqs to the turn numbers the memory was derived from.\n"
    "- Set confidence in [0.0, 1.0] to how strongly the transcript supports the "
    "claim: ~0.9+ for something the user explicitly stated or confirmed, ~0.6-0.8 "
    "for a clearly implied claim, lower when it is an uncertain inference. Omit it "
    "only when you genuinely cannot judge.\n"
    "- Return an empty array when nothing is worth remembering.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object matching this JSON Schema:\n\n"
    + json.dumps(_EXTRACTION_SCHEMA, indent=2)
)


def _build_system_prompt(domain_fact_guidance: str | None = None) -> str:
    """Assemble the extraction system prompt, optionally scoped to a domain.

    ``domain_fact_guidance`` is an *additive* slot inserted between the kind
    definitions and the rules: it narrows what subject matter counts as a
    durable ``fact``/``correction`` for one application, but is deliberately
    placed *above* the provenance rule and framed as topic-scoping only, so it
    cannot relax the rule that a subject-matter fact is durable only when the
    USER is its source.  Empty/omitted reproduces the generic prompt.
    """
    domain_block = ""
    if domain_fact_guidance and domain_fact_guidance.strip():
        domain_block = (
            "Subject-matter scope for this application — treat a `fact` or "
            "`correction` as durable only when it is knowledge of the kind "
            "described here. This narrows the topic; it does NOT relax the "
            "provenance rule below (the USER must still be the source):\n"
            + domain_fact_guidance.strip()
            + "\n\n"
        )
    return _PROMPT_INTRO_AND_KINDS + domain_block + _PROMPT_RULES_AND_SCHEMA


# Default (generic) prompt, used when no domain guidance is supplied.
_SYSTEM_PROMPT = _build_system_prompt()


def _thread_observation_date(
    thread: list[MemoryMessage], seq_to_event: dict[int, MemoryEvent]
) -> datetime | None:
    """When the turns being distilled took place — the anchor for relative refs.

    Distillation runs offline, often long after the session settled, so "today"
    at distill time is the wrong anchor for a turn that said "yesterday".  We pin
    relative references to *when the conversation happened*.

    Crucially this anchors on the turns *in this pass* — the post-watermark
    ``thread`` — not the original session start.  A session resumed days later has
    its earlier turns already distilled (the watermark advanced), so this pass
    sees only the new turns; anchoring on the original ``session_started`` would
    resolve the resumed turns' "yesterday" against the wrong day.  ``thread`` is
    seq-ordered, so the first turn's timestamp is when this batch began.

    Returns ``None`` only when no thread turn maps to a timestamped event (no
    anchor to offer); a single observation date still suits the whole batch, so we
    do not thread per-turn timestamps into the transcript.
    """
    for m in thread:
        event = seq_to_event.get(m.seq)
        if event is not None and event.created_at is not None:
            return event.created_at
    return None


def _observation_header(observation_date: datetime | None) -> str:
    """The transcript preamble that anchors relative time references.

    Empty when there is no date to offer, so the transcript is unchanged and the
    prompt's temporal rule simply has nothing to resolve against.
    """
    if observation_date is None:
        return ""
    return (
        "Observation date (when this conversation took place): "
        f"{observation_date:%Y-%m-%d (%A)}.\n"
        "Resolve relative time references in the transcript against this date.\n\n"
    )


def _existing_memories_block(records: list[LongTermRecord]) -> str:
    """The transcript suffix listing already-captured memories for reconcile + linking.

    Front-loading the related existing memories into the extraction prompt lets
    the extractor, in the single call, (a) reconcile each new memory against the
    store — emit an ``operation`` (ADD / UPDATE / DELETE / NOOP) and a
    ``target_id`` against these records — so a duplicate becomes an in-place
    UPDATE instead of being caught (after an embed + a separate reconcile LLM
    call) downstream; and (b) emit ``linked_memory_ids`` edges from a new memory
    to the related existing ones, building the memory graph that recall
    traverses.  Empty when there are no related records, so the prompt's
    reconcile/linking rules simply have nothing to match against.

    Records are tagged with their *position* (``[id=0]``, ``[id=1]``…), not their
    real ``memory_id``: an LLM asked to echo a 36-char UUID eventually invents or
    typos one (the same anti-hallucination masking :meth:`LongTermMemory._decide`
    uses).  The position maps back to the real id in :meth:`Distiller._build_candidate`.
    """
    if not records:
        return ""
    lines = "\n".join(f"- [id={i}] {r.content}" for i, r in enumerate(records))
    return (
        "\n\n## Existing memories (already captured; for reconciliation and linking)\n"
        + lines
    )


class Distiller:
    """Promotes durable records out of one session log on settle.

    Args:
        episodic:  The append-only event log to replay.
        long_term: The service candidates are reconciled into.
        llm:       Extraction LLM (the candidate-generation step).
        max_retries: Retries on an unparseable/invalid extraction response.
        min_confidence: Per-kind confidence floor; candidates scored below their
            kind's floor (or without a usable score) are abandoned before
            reconcile (see :data:`DEFAULT_MIN_CONFIDENCE`).
        domain_fact_guidance: Optional application-specific description of which
            subject-matter facts are durable, injected as an additive topic
            scope in the extraction prompt (see :func:`_build_system_prompt`).
            ``None`` uses the generic prompt.
        existing_memory_limit: How many related existing memories to vector-recall
            and inject into the extraction prompt as a reconcile + linking
            reference (see :func:`_existing_memories_block`).  ``0`` disables the
            lookup and reproduces the blind-extract behaviour.
        single_call: When True (default), the extractor decides each candidate's
            reconcile op in the single extraction call and
            :meth:`LongTermMemory.reconcile_decided` just applies it — one LLM
            call per session.  When False, fall back to the auditable per-candidate
            :meth:`LongTermMemory.reconcile`, which re-recalls and spends one more
            LLM call per candidate (N+1 calls per session).
        auto_link_max_entity_ratio: Deterministic link augmentation (see
            :meth:`_auto_link`).  A candidate is auto-linked to a recalled existing
            memory when they share an entity present in no more than this fraction
            of active records — discriminative enough to be a real signal.  Common
            entities (recurring speakers) exceed the ratio and are skipped, so the
            graph never collapses into a same-subject clique.  ``0`` disables it and
            leaves edges to the LLM alone.
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        long_term: LongTermMemory,
        llm: LLMBase,
        *,
        max_retries: int = 2,
        min_confidence: dict[MemoryKind, float] | None = None,
        domain_fact_guidance: str | None = None,
        existing_memory_limit: int = 10,
        single_call: bool = True,
        auto_link_max_entity_ratio: float = 0.1,
    ) -> None:
        self._episodic = episodic
        self._long_term = long_term
        self._llm = llm
        self._max_retries = max_retries
        self._min_confidence = min_confidence or DEFAULT_MIN_CONFIDENCE
        self._system_prompt = _build_system_prompt(domain_fact_guidance)
        self._existing_memory_limit = existing_memory_limit
        self._single_call = single_call
        self._auto_link_max_entity_ratio = auto_link_max_entity_ratio

    async def distill_session(self, *, session_id: str) -> list[str]:
        """Replay, extract candidates, reconcile each; return affected memory ids.

        Idempotent across re-runs: a ``session_distilled`` watermark records how
        far prior passes extracted, so only turns past it are projected.  Without
        it a re-distill (sessions are resumable / re-closable) would re-extract
        the whole transcript and reinforce every already-captured record, drifting
        confidence toward 1.0 with no new evidence.
        """
        events = await self._episodic.replay(session_id=session_id)
        app_id = next((e.app_id for e in events if e.app_id), None)
        distilled_through = latest_distillation(events)
        thread = project_thread(events, since_seq=distilled_through)
        if not thread:
            logger.info(
                "[distill] app=%s session=%s no turns past distilled_through=%d; "
                "nothing to distill",
                app_id, session_id, distilled_through,
            )
            return []
        logger.info(
            "[distill] app=%s session=%s distilling %d turn(s) past "
            "distilled_through=%d from %d event(s)",
            app_id, session_id, len(thread), distilled_through, len(events),
        )

        transcript = "\n".join(
            f"[{m.seq} {m.role.value}] {m.content}" for m in thread
        )
        seq_to_ref = {e.seq: e.ref for e in events}
        seq_to_event = {e.seq: e for e in events}
        observation_date = _thread_observation_date(thread, seq_to_event)
        if observation_date is None:
            # Invariant: ``thread`` is non-empty here (we returned above otherwise)
            # and every turn is projected from a timestamped log event, so an anchor
            # must exist.  ``None`` means a turn reached the thread without a backing
            # event — a bug we surface rather than promote an undated memory.
            raise RuntimeError(
                f"no observation anchor for session={session_id} despite "
                f"{len(thread)} turn(s); a thread turn lacks a timestamped event"
            )
        existing = await self._recall_existing(transcript)
        parsed = await self._extract(
            transcript, observation_date=observation_date, existing=existing
        )
        if not parsed:
            return []

        # Position -> real memory_id, mirroring the masked ids in the existing-
        # memories block, to resolve the extractor's linked_memory_ids.
        link_id_map = {i: r.memory_id for i, r in enumerate(existing)}

        candidates: list[MemoryCandidate] = []
        for item in parsed.get("memories", []):
            candidate = self._build_candidate(
                item, seq_to_ref, seq_to_event, link_id_map, observation_date
            )
            if candidate is not None:
                candidates.append(candidate)

        # Deterministic edge augmentation: link candidates to recalled existing
        # memories that share a discriminative entity, on top of whatever edges the
        # LLM emitted.  No-op when disabled or when there is nothing to link to.
        await self._auto_link(candidates, existing)

        # Reconcile is order-dependent (each candidate sees what prior ones
        # wrote), so it stays a sequential loop — but every candidate's content
        # is embedded the same way regardless of order, so embed them all in one
        # call up front and thread the cache through.
        embeddings = await self._long_term.embed_contents(candidates)

        # Single-call path: the extractor already decided each op, so just apply
        # it (no per-candidate recall/LLM call).  Otherwise fall back to the
        # auditable per-candidate reconcile.
        reconcile = (
            self._long_term.reconcile_decided
            if self._single_call
            else self._long_term.reconcile
        )
        memory_ids: list[str] = []
        for candidate in candidates:
            try:
                memory_ids.append(
                    await reconcile(candidate=candidate, embeddings=embeddings)
                )
            except Exception:
                logger.warning(
                    "[distill] app=%s reconcile failed for a candidate in session=%s",
                    app_id, session_id, exc_info=True,
                )
        logger.info(
            "[distill] app=%s session=%s extracted=%d reconciled=%d",
            app_id, session_id, len(parsed.get("memories", [])), len(memory_ids),
        )
        # Advance the watermark past every turn this pass examined — even when it
        # extracted nothing (those turns were judged and produced no memory).  Only
        # a successful extraction reaches here: an unparseable/failed ``_extract``
        # returned ``None`` above, so its turns stay un-watermarked and are retried.
        await self._record_watermark(
            session_id=session_id,
            distilled_through=max(m.seq for m in thread),
            memory_count=len(memory_ids),
            app_id=app_id,
        )
        return memory_ids

    async def _record_watermark(
        self,
        *,
        session_id: str,
        distilled_through: int,
        memory_count: int,
        app_id: str | None,
    ) -> None:
        """Append + flush the ``session_distilled`` watermark for this pass.

        Best-effort and symmetric with compaction's ``replaces_through``: the
        distiller is the sole writer of a settled session, so it appends one
        ``session_distilled`` event recording the last turn it extracted through
        and flushes it durably.  A failure here does not undo the records already
        promoted — it only risks a future re-distill re-examining these turns — so
        it is logged, not raised (the records are the durable outcome, the
        watermark only an optimization/guard).
        """
        try:
            await self._episodic.record_distillation(
                session_id=session_id,
                distilled_through=distilled_through,
                memory_count=memory_count,
            )
            await self._episodic.flush(session_id=session_id)
        except Exception:
            logger.warning(
                "[distill] app=%s session=%s failed to record distilled_through=%d "
                "watermark; a later distill may re-examine these turns",
                app_id, session_id, distilled_through, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_candidate(
        self,
        item: dict,
        seq_to_ref: dict[int, EventRef],
        seq_to_event: dict[int, MemoryEvent],
        link_id_map: dict[int, str],
        observation_date: datetime,
    ) -> MemoryCandidate | None:
        try:
            kind = MemoryKind(item["kind"])
        except (KeyError, ValueError):
            logger.warning("[distill] dropping candidate with bad kind: %s", item)
            return None
        content = (item.get("content") or "").strip()
        if not content:
            return None

        confidence = self._parse_confidence(item.get("confidence"))
        floor = self._min_confidence.get(kind, _FALLBACK_MIN_CONFIDENCE)
        if confidence is None or confidence < floor:
            logger.info(
                "[distill] dropping low/unscored %s candidate "
                "(confidence=%s < floor=%.2f): %.80s",
                kind.value, confidence, floor, content,
            )
            return None

        source_seqs = [s for s in item.get("source_seqs", []) if s in seq_to_ref]
        source_event_ids = [seq_to_ref[s] for s in source_seqs]
        snapshot = self._snapshot(source_seqs, seq_to_event)
        observed_at = self._observed_at(source_seqs, seq_to_event, observation_date)
        operation, target_memory_id, revised_content = self._resolve_operation(
            item, link_id_map, content
        )
        return MemoryCandidate(
            content=content,
            kind=kind,
            entities=normalize_entities(item.get("entities", [])),
            linked_memory_ids=self._resolve_links(
                item.get("linked_memory_ids", []), link_id_map
            ),
            source_event_ids=source_event_ids,
            evidence_snapshot=snapshot,
            observed_at=observed_at,
            confidence=confidence,
            operation=operation,
            target_memory_id=target_memory_id,
            revised_content=revised_content,
        )

    @staticmethod
    def _resolve_operation(
        item: dict, link_id_map: dict[int, str], content: str
    ) -> tuple[ReconcileOp, str | None, str | None]:
        """Lift the extractor's single-call reconcile decision off one item.

        Returns ``(operation, target_memory_id, revised_content)`` ready for
        :meth:`LongTermMemory.reconcile_decided`.  The extractor names its target
        by the masked id from the `## Existing memories` block, mapped back to a
        real ``memory_id`` via ``link_id_map`` (the same namespace as
        ``linked_memory_ids``).  Anything that can't be a clean non-ADD decision
        degrades to ADD, so a malformed/hallucinated op or an unresolvable target
        promotes the candidate as new rather than mis-targeting a record:

        - an unparseable or unknown ``operation`` -> ADD;
        - an UPDATE/DELETE/NOOP whose ``target_id`` doesn't resolve to a shown id
          -> ADD (it named a record we never offered).

        ``revised_content`` is kept only for an UPDATE that actually rewords the
        claim (non-empty and different from ``content``); it is irrelevant to the
        other ops.
        """
        try:
            operation = ReconcileOp(item.get("operation", ReconcileOp.ADD.value))
        except ValueError:
            return ReconcileOp.ADD, None, None
        if operation is ReconcileOp.ADD:
            return ReconcileOp.ADD, None, None

        target_memory_id = link_id_map.get(_as_index(item.get("target_id")))
        if target_memory_id is None:
            logger.info(
                "[distill] %s names unresolvable target_id=%r; degrading to ADD: %.80s",
                operation.value, item.get("target_id"), content,
            )
            return ReconcileOp.ADD, None, None

        revised_content = None
        if operation is ReconcileOp.UPDATE:
            revised = (item.get("revised_content") or "").strip()
            if revised and revised != content:
                revised_content = revised
        return operation, target_memory_id, revised_content

    @staticmethod
    def _resolve_links(
        raw: object, link_id_map: dict[int, str]
    ) -> list[str]:
        """Map the extractor's masked link ids back to real ``memory_id``s.

        Drops anything that isn't an index we showed (an out-of-range or
        hallucinated value), order-preserving and deduped, so a bad id degrades
        to "no edge" rather than a dangling reference.
        """
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for value in raw:
            index = _as_index(value)
            if index is None:
                continue
            mid = link_id_map.get(index)
            if mid is not None and mid not in out:
                out.append(mid)
        return out

    @staticmethod
    def _parse_confidence(value: object) -> float | None:
        """Clamp the LLM's confidence to [0, 1]; ``None`` if not a usable number.

        A non-numeric value signals an unreliable extraction; the caller
        abandons such a candidate rather than salvaging it with a default.
        """
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _snapshot(
        source_seqs: list[int], seq_to_event: dict[int, MemoryEvent]
    ) -> dict:
        """Copy the deciding turns' text (and any cited evidence) into the record.

        Self-containment: the snapshot keeps the record valid after the source
        session log expires (docs/long-term-memory.md#pipeline, step 5).
        """
        turns: list[dict] = []
        cited: list[dict] = []
        for seq in source_seqs:
            event = seq_to_event.get(seq)
            if event is None:
                continue
            turns.append(
                {
                    "seq": seq,
                    "type": event.event_type.value,
                    "text": event.payload.get("text", ""),
                }
            )
            # A final_answer carries the evidence it cited; copy resolvable
            # cited text so the snapshot stands alone.
            if event.event_type is EventType.FINAL_ANSWER:
                for ref in event.payload.get("cited_ids", []):
                    ref_seq = ref.get("seq") if isinstance(ref, dict) else None
                    cited_event = seq_to_event.get(ref_seq) if ref_seq is not None else None
                    if cited_event is not None:
                        cited.append(
                            {
                                "seq": ref_seq,
                                "type": cited_event.event_type.value,
                                "payload": Distiller._truncate_payload(cited_event),
                            }
                        )
        snapshot: dict = {"turns": turns}
        if cited:
            snapshot["cited"] = cited
        return snapshot

    @staticmethod
    def _observed_at(
        source_seqs: list[int],
        seq_to_event: dict[int, MemoryEvent],
        fallback: datetime,
    ) -> datetime:
        """When the claim was observed: the latest of its source turns' timestamps.

        Dating a memory by *its own* source turns rather than one session-level
        anchor keeps it correct when a session is long or resumed days later — a
        fact asserted late in the transcript is dated by that turn, not by session
        start (see LongTermRecord.observed_at).  The latest source turn is the most
        recent assertion of the claim.  Falls back to ``fallback`` (the session
        observation date, itself always present — see ``distill_session``) when the
        candidate cites no resolvable source turn, so the result is never None."""
        stamps = [
            e.created_at
            for s in source_seqs
            if (e := seq_to_event.get(s)) is not None and e.created_at is not None
        ]
        return max(stamps) if stamps else fallback

    @staticmethod
    def _truncate_payload(event: MemoryEvent) -> dict:
        """Cap a cited payload's serialized size before it lands in a snapshot.

        A final_answer can cite large tool results; copied wholesale they bloat
        the evidence_snapshot JSON column.  When a payload exceeds
        ``_MAX_CITED_PAYLOAD_BYTES`` it is replaced with a truncated stand-in
        that preserves a head of the original text and records what was dropped.
        """
        payload = event.payload
        size = len(json.dumps(payload, default=str).encode("utf-8"))
        if size <= _MAX_CITED_PAYLOAD_BYTES:
            return payload
        logger.warning(
            "[distill] cited payload too large, truncating: app=%s session=%s "
            "seq=%s size=%d limit=%d",
            event.app_id,
            event.session_id,
            event.seq,
            size,
            _MAX_CITED_PAYLOAD_BYTES,
        )
        text = payload.get("text")
        if not isinstance(text, str):
            text = json.dumps(payload, default=str)
        head = text.encode("utf-8")[:_MAX_CITED_PAYLOAD_BYTES].decode(
            "utf-8", "ignore"
        )
        return {
            "_truncated": True,
            "_original_bytes": size,
            "text": head,
        }

    async def _auto_link(
        self,
        candidates: list[MemoryCandidate],
        existing: list[LongTermRecord],
    ) -> None:
        """Add deterministic edges from candidates to related existing memories.

        Mutates each candidate's ``linked_memory_ids`` in place, augmenting the
        LLM-emitted edges.  A candidate is linked to an existing record when they
        share at least one *discriminative* entity — one present in no more than
        ``auto_link_max_entity_ratio`` of all active records.  Ubiquitous entities
        (a recurring speaker who tags most records) exceed the ratio and are
        ignored, so this never wires every same-subject record into a clique; rare,
        specific entities ("homeless shelter", "career fair") are what create
        edges.  Each candidate gains at most ``_MAX_AUTO_LINKS`` new edges.

        Best-effort: a frequency-scan failure leaves the LLM's edges untouched
        rather than sinking the distillation.
        """
        if self._auto_link_max_entity_ratio <= 0 or not candidates or not existing:
            return
        try:
            counts, total = await self._long_term.active_entity_frequencies()
        except Exception:
            logger.warning(
                "[distill] entity-frequency scan failed; skipping auto-link",
                exc_info=True,
            )
            return
        if total <= 0:
            return
        cap = self._auto_link_max_entity_ratio * total

        # Pre-normalize each existing record's discriminative entities once.
        existing_disc: list[tuple[str, set[str]]] = []
        for record in existing:
            disc = {
                e
                for e in normalize_entities(record.entities)
                if counts.get(e, 0) <= cap
            }
            if disc:
                existing_disc.append((record.memory_id, disc))
        if not existing_disc:
            return

        added = 0
        for candidate in candidates:
            cand_disc = {
                e
                for e in normalize_entities(candidate.entities)
                if counts.get(e, 0) <= cap
            }
            if not cand_disc:
                continue
            existing_links = set(candidate.linked_memory_ids)
            new_links: list[str] = []
            for memory_id, disc in existing_disc:
                if len(new_links) >= _MAX_AUTO_LINKS:
                    break
                if memory_id in existing_links:
                    continue
                if cand_disc & disc:
                    new_links.append(memory_id)
            if new_links:
                candidate.linked_memory_ids = list(
                    dict.fromkeys(candidate.linked_memory_ids + new_links)
                )
                added += len(new_links)
        if added:
            logger.info(
                "[distill] auto-linked %d edge(s) across %d candidate(s) "
                "(entity ratio<=%.2f of %d active record(s))",
                added, len(candidates), self._auto_link_max_entity_ratio, total,
            )

    async def _recall_existing(self, transcript: str) -> list[LongTermRecord]:
        """Vector-recall the active memories most related to this session.

        Front-loaded into the extraction prompt so the extractor dedups against
        accumulated belief up front (see :func:`_existing_memories_block`).  The
        whole transcript is the recall query — the same semantic surface the
        candidates will be drawn from.  Best-effort: a recall failure must not
        sink the distillation, so it degrades to no context (blind extract).
        """
        if self._existing_memory_limit <= 0:
            return []
        try:
            return await self._long_term.recall(
                query=transcript, limit=self._existing_memory_limit
            )
        except Exception:
            logger.warning(
                "[distill] existing-memory recall failed; extracting without it",
                exc_info=True,
            )
            return []

    async def _extract(
        self,
        transcript: str,
        *,
        observation_date: datetime | None = None,
        existing: list[LongTermRecord] | None = None,
    ) -> dict | None:
        """LLM extraction → parsed+validated JSON, retrying (extraction pattern)."""
        user_content = (
            _observation_header(observation_date)
            + transcript
            + _existing_memories_block(existing or [])
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        content = ""
        for attempt in range(self._max_retries + 1):
            try:
                result = await self._llm.complete(messages, model="mini")
                content = (result.get("content") or "").strip()
                parsed = json.loads(_strip_code_fence(content))
                memories = _coerce_memories(parsed)
                if memories is None:
                    raise ValueError("no `memories` array in extraction output")
                normalized = {"memories": memories}
                jsonschema.validate(instance=normalized, schema=_EXTRACTION_SCHEMA)
                logger.debug("[distill] parsed=%s messages=%s", normalized, messages)
                return normalized
            except (json.JSONDecodeError, jsonschema.ValidationError, ValueError):
                logger.error("[distill] extraction JSON invalid, attempt=%d content=%s", attempt, content)
                if attempt < self._max_retries:
                    continue
                logger.warning("[distill] extraction JSON invalid after retries", exc_info=True)
                return None
            except Exception:
                logger.warning("[distill] extraction LLM call failed", exc_info=True)
                return None
        return None
