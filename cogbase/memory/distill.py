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
)
from cogbase.memory.projection import project_thread

logger = logging.getLogger(__name__)

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
                    "source_seqs": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["content", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You distill durable, long-term memories from a conversation transcript "
    "between a user and an assistant.  Each transcript line is prefixed with its "
    "turn number, e.g. [3 user] or [4 assistant].\n\n"
    "Extract only knowledge worth remembering across future sessions:\n"
    "- preference: a stable interaction preference (e.g. 'prefers concise "
    "answers', 'always wants citations').\n"
    "- fact: a stable fact about the user, their project, or their organization.\n"
    "- correction: a fact the user explicitly corrected or confirmed.\n"
    "- retrieval_hint: a recurring intent-to-tool-chain or routing pattern worth "
    "recalling at query time.\n\n"
    "Rules:\n"
    "- Do NOT extract ephemeral, one-off, or task-local details.\n"
    "- Write each memory as a single self-contained natural-language claim.\n"
    "- Set source_seqs to the turn numbers the memory was derived from.\n"
    "- Return an empty array when nothing is worth remembering.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object matching this JSON Schema:\n\n"
    + json.dumps(_EXTRACTION_SCHEMA, indent=2)
)


class Distiller:
    """Promotes durable records out of one session log on settle.

    Args:
        episodic:  The append-only event log to replay.
        long_term: The service candidates are reconciled into.
        llm:       Extraction LLM (the candidate-generation step).
        max_retries: Retries on an unparseable/invalid extraction response.
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        long_term: LongTermMemory,
        llm: LLMBase,
        *,
        max_retries: int = 2,
    ) -> None:
        self._episodic = episodic
        self._long_term = long_term
        self._llm = llm
        self._max_retries = max_retries

    async def distill_session(self, *, session_id: str) -> list[str]:
        """Replay, extract candidates, reconcile each; return affected memory ids."""
        events = await self._episodic.replay(session_id=session_id)
        thread = project_thread(events)
        if not thread:
            logger.info("[distill] session=%s has no thread; nothing to distill", session_id)
            return []

        transcript = "\n".join(
            f"[{m.seq} {m.role.value}] {m.content}" for m in thread
        )
        parsed = await self._extract(transcript)
        if not parsed:
            return []

        seq_to_ref = {e.seq: e.ref for e in events}
        seq_to_event = {e.seq: e for e in events}

        memory_ids: list[str] = []
        for item in parsed.get("memories", []):
            candidate = self._build_candidate(item, seq_to_ref, seq_to_event)
            if candidate is None:
                continue
            try:
                memory_ids.append(
                    await self._long_term.reconcile(candidate=candidate)
                )
            except Exception:
                logger.warning(
                    "[distill] reconcile failed for a candidate in session=%s",
                    session_id, exc_info=True,
                )
        logger.info(
            "[distill] session=%s extracted=%d reconciled=%d",
            session_id, len(parsed.get("memories", [])), len(memory_ids),
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

        source_seqs = [s for s in item.get("source_seqs", []) if s in seq_to_ref]
        source_event_ids = [seq_to_ref[s] for s in source_seqs]
        snapshot = self._snapshot(source_seqs, seq_to_event)
        return MemoryCandidate(
            content=content,
            kind=kind,
            source_event_ids=source_event_ids,
            evidence_snapshot=snapshot,
        )

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

    async def _extract(self, transcript: str) -> dict | None:
        """LLM extraction → parsed+validated JSON, retrying (extraction pattern)."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        for attempt in range(self._max_retries + 1):
            try:
                result = await self._llm.complete(messages, model="mini")
                content = (result.get("content") or "").strip()
                parsed = json.loads(content)
                jsonschema.validate(instance=parsed, schema=_EXTRACTION_SCHEMA)
                return parsed
            except (json.JSONDecodeError, jsonschema.ValidationError):
                if attempt < self._max_retries:
                    continue
                logger.warning("[distill] extraction JSON invalid after retries", exc_info=True)
                return None
            except Exception:
                logger.warning("[distill] extraction LLM call failed", exc_info=True)
                return None
        return None
