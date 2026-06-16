"""Distiller: the offline pipeline that promotes durable memory from logs.

Distillation is ``extract-structured`` pointed at session logs instead of
documents (see docs/long-term-memory.md): a session log is the "document" and
long-term records are the extracted "structured records".  It reuses the
extraction *pattern* (schema-validated JSON + retry) but not the
``Document`` / ``doc_id`` contract — output feeds :meth:`LongTermMemory.reconcile`,
not an upsert by primary key.

The flow (docs/long-term-memory-implementation-plan.md, Step 3):

1. :meth:`EpisodicMemory.replay` the whole session log (it is short and bounded).
2. Project the conversational thread (the shared
   :mod:`cogbase.memory.projection` helper) and resolve provenance.
3. LLM-extract candidate memories, each carrying the ``source_event_ids`` it was
   derived from and a snapshot of the deciding turns for self-containment.
4. :meth:`LongTermMemory.reconcile` each candidate; return the affected ids.

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
    MemoryCandidate,
    MemoryEvent,
    MemoryKind,
    normalize_entities,
)
from cogbase.memory.projection import project_thread

logger = logging.getLogger(__name__)

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
    "- Do NOT extract trivial chit-chat or transient task mechanics.\n"
    "- A subject-matter fact is durable only when the USER is its source — they "
    "asserted it, corrected the assistant, or confirmed a claim. Do NOT distill a "
    "fact that originates solely in an assistant turn derived from document "
    "retrieval; the source documents already own that knowledge.\n"
    "- Write each memory as a single self-contained natural-language claim.\n"
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


def _session_observation_date(events: list[MemoryEvent]) -> datetime | None:
    """When the conversation took place — the anchor for relative time refs.

    Distillation runs offline, often long after the session settled, so "today"
    at distill time is the wrong anchor for a turn that said "yesterday".  We
    pin every relative reference to *when the conversation happened*: the
    ``session_started`` event if present, else the earliest event's timestamp.
    Returns ``None`` only for an empty log (no anchor to offer).
    """
    started = next(
        (e for e in events if e.event_type is EventType.SESSION_STARTED), None
    )
    if started is not None:
        return started.created_at
    return min((e.created_at for e in events), default=None)


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
    ) -> None:
        self._episodic = episodic
        self._long_term = long_term
        self._llm = llm
        self._max_retries = max_retries
        self._min_confidence = min_confidence or DEFAULT_MIN_CONFIDENCE
        self._system_prompt = _build_system_prompt(domain_fact_guidance)

    async def distill_session(self, *, session_id: str) -> list[str]:
        """Replay, extract candidates, reconcile each; return affected memory ids."""
        events = await self._episodic.replay(session_id=session_id)
        app_id = next((e.app_id for e in events if e.app_id), None)
        thread = project_thread(events)
        if not thread:
            logger.info(
                "[distill] app=%s session=%s has no thread; nothing to distill",
                app_id, session_id,
            )
            return []
        logger.info(
            "[distill] app=%s session=%s distilling %d turn(s) from %d event(s)",
            app_id, session_id, len(thread), len(events),
        )

        transcript = "\n".join(
            f"[{m.seq} {m.role.value}] {m.content}" for m in thread
        )
        observation_date = _session_observation_date(events)
        parsed = await self._extract(transcript, observation_date=observation_date)
        if not parsed:
            return []

        seq_to_ref = {e.seq: e.ref for e in events}
        seq_to_event = {e.seq: e for e in events}

        candidates: list[MemoryCandidate] = []
        for item in parsed.get("memories", []):
            candidate = self._build_candidate(item, seq_to_ref, seq_to_event)
            if candidate is not None:
                candidates.append(candidate)

        # Reconcile is order-dependent (each candidate sees what prior ones
        # wrote), so it stays a sequential loop — but every candidate's content
        # is embedded the same way regardless of order, so embed them all in one
        # call up front and thread the cache through.
        embeddings = await self._long_term.embed_contents(candidates)

        memory_ids: list[str] = []
        for candidate in candidates:
            try:
                memory_ids.append(
                    await self._long_term.reconcile(
                        candidate=candidate, embeddings=embeddings
                    )
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
        return memory_ids

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_candidate(
        self,
        item: dict,
        seq_to_ref: dict[int, EventRef],
        seq_to_event: dict[int, MemoryEvent],
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
        return MemoryCandidate(
            content=content,
            kind=kind,
            entities=normalize_entities(item.get("entities", [])),
            source_event_ids=source_event_ids,
            evidence_snapshot=snapshot,
            confidence=confidence,
        )

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
                                "payload": cited_event.payload,
                            }
                        )
        snapshot: dict = {"turns": turns}
        if cited:
            snapshot["cited"] = cited
        return snapshot

    async def _extract(
        self, transcript: str, *, observation_date: datetime | None = None
    ) -> dict | None:
        """LLM extraction → parsed+validated JSON, retrying (extraction pattern)."""
        user_content = _observation_header(observation_date) + transcript
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
