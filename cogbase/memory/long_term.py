"""Long-term memory: the online recall / reconcile / promote service.

``LongTermMemory`` is the curated, durable knowledge tier (see
docs/long-term-memory.md).  It sits over a structured store (canonical records)
and a vector store (semantic recall over ``content`` and the reconcile step's
"find related records" lookup).  Both stores are passed in **already
app-scoped** by the factory, so this service treats the app partition as given
— it is the only partition until multi-user / RBAC is designed.

Four operations make up the surface:

- :meth:`recall` — online, on the query path: a vector search over memory
  ``content``, filtered to active records.  Results are marked
  memory-derived by the caller so the evidence policy can keep them distinct
  from document-backed claims.
- :meth:`lookup` — the pull counterpart to :meth:`recall`, backing the query
  runner's ``memory_lookup`` tool: an optional semantic query combined with
  ``kind`` / ``entities`` filters, for when the per-turn injection isn't enough
  ("what do you know about project X?").
- :meth:`promote` — the ADD path: embed, write the structured record and its
  content vector with a provenance snapshot.  Behaviour-affecting kinds land at
  ``pending_review``; :meth:`list_pending` / :meth:`review_many` are the gate
  that promotes them to ``active`` (or supersedes them on reject).
- :meth:`reconcile` — **the crux, no analog in the document pipeline.**  Merge a
  new observation against accumulated belief: vector-recall related
  records, let an LLM emit one ``ReconcileOp`` (ADD / UPDATE / DELETE / NOOP),
  and apply it (reinforce confidence, revise content, or supersede a
  contradicted record).

Unlike the episodic log (append-only) this store is mutable: reconciliation
updates and supersedes records in place so it stays curated, not append-only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import jsonschema

from cogbase.core.models import Chunk
from cogbase.embeddings.base import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.memory.models import (
    EventRef,
    LongTermRecord,
    MemoryCandidate,
    MemoryKind,
    MemoryStatus,
    ReconcileOp,
    ReviewDecision,
    ReviewOutcome,
    ReviewResult,
    normalize_entities,
)
from cogbase.stores.filters import Col
from cogbase.stores.vector.base import VectorCollectionSchema, VectorStoreBase
from cogbase.stores.structured.base import StructuredStoreBase

logger = logging.getLogger(__name__)

DEFAULT_STRUCTURED_COLLECTION = "long_term_memory"
DEFAULT_VECTOR_COLLECTION = "long_term_memory_content"

# How much an UPDATE (reinforce) bumps confidence, capped at 1.0.
CONFIDENCE_REINFORCE = 0.1

# Promotion gate by kind + confidence (docs/long-term-memory.md#promotion-confidence-and-review):
# a record auto-promotes to ``active`` only when its confidence is at/above its
# kind's threshold here, otherwise it waits at ``pending_review`` for a reviewer.
# Low-risk interaction signals (preferences, hints) use a 0.0 threshold so they
# always auto-promote; a fact auto-promotes only when strongly supported; a
# correction overrides existing belief, so its threshold is unreachable (>1.0) —
# it always waits for review regardless of how confident the LLM is.
AUTO_PROMOTE_CONFIDENCE: dict[MemoryKind, float] = {
    MemoryKind.PREFERENCE: 0.0,
    MemoryKind.RETRIEVAL_HINT: 0.0,
    MemoryKind.FACT: 0.85,
    MemoryKind.CORRECTION: 1.01,
}

# Threshold for a kind missing from the table (defensive): hold for review.
_FALLBACK_AUTO_PROMOTE = 1.01

# How many related records to surface to the reconcile LLM.
_RECONCILE_CANDIDATES = 5

# Upper bound on a single review batch — the loop issues two store writes per
# accepted/rejected record, so cap it to keep one call's fan-out bounded.
MAX_REVIEW_BATCH = 500

_RECONCILE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
        "target_memory_id": {"type": ["integer", "string", "null"]},
        "revised_content": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
    },
    "required": ["operation"],
    "additionalProperties": False,
}

_RECONCILE_INTRO_AND_OPS = (
    "You curate a long-term memory store.  You are given a NEW observation and a "
    "list of EXISTING related memories.  Decide how the new "
    "observation reconciles against accumulated belief, and return exactly one "
    "operation as a JSON object.\n\n"
    "Each EXISTING memory is labelled with a small integer id.  When an operation "
    "refers to one, set target_memory_id to that integer — do NOT invent or echo "
    "any other identifier.\n\n"
    "Operations:\n"
    "- ADD: the observation is genuinely new; no existing memory covers it.\n"
    "- UPDATE: an existing memory already says this — reinforce it, or refine its "
    "wording.  Set target_memory_id to that memory's id; set revised_content only "
    "if the wording should change.\n"
    "- DELETE: the observation CONTRADICTS an existing memory and should replace "
    "it.  Set target_memory_id to the contradicted memory's id (it will be "
    "superseded and the new observation promoted in its place).\n"
    "- NOOP: already fully known and correctly stated; nothing to change.\n\n"
)

_RECONCILE_RULES_AND_SCHEMA = (
    "Rules:\n"
    "- Prefer UPDATE over ADD when an existing memory expresses the same claim.\n"
    "- Use DELETE only for a real contradiction, not a mere refinement.\n"
    "- Return ONLY the JSON object — no explanation outside it, no markdown fences.\n\n"
    "Return a single JSON object matching this JSON Schema:\n\n"
    + json.dumps(_RECONCILE_SCHEMA, indent=2)
)


def _build_reconcile_prompt(reconcile_guidance: str | None = None) -> str:
    """Assemble the reconcile system prompt, optionally scoped to a domain.

    The consolidation-side analog of distill's ``domain_fact_guidance`` (and of
    mem0's ``custom_update_memory_prompt``): ``reconcile_guidance`` is an
    *additive* slot inserted between the operation definitions and the rules.  It
    lets one application inject domain judgement about *when* two observations are
    the same claim, a genuine contradiction, or merely a refinement — e.g.
    "two contract clauses with different effective dates are distinct records,
    not a contradiction" — without being able to change the ADD/UPDATE/DELETE/
    NOOP operation set or the JSON output format, which the rules below fix.
    Empty/omitted reproduces the generic prompt.
    """
    guidance_block = ""
    if reconcile_guidance and reconcile_guidance.strip():
        guidance_block = (
            "Domain reconciliation guidance for this application — weigh it when "
            "choosing the operation (in particular, what counts as the same claim "
            "versus a genuine contradiction). It adds domain judgement; it does "
            "NOT change the operation set or the output format below:\n"
            + reconcile_guidance.strip()
            + "\n\n"
        )
    return _RECONCILE_INTRO_AND_OPS + guidance_block + _RECONCILE_RULES_AND_SCHEMA


# Default (generic) prompt, used when no domain guidance is supplied.
_RECONCILE_SYSTEM_PROMPT = _build_reconcile_prompt()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LongTermMemory:
    """Recall / reconcile / promote over app-scoped structured + vector stores.

    Args:
        structured_store: Canonical record store, **already app-scoped**.
        vector_store:      Semantic-recall store over ``content``, **already
                           app-scoped**.
        llm:               Used by :meth:`reconcile` to decide the operation.
        embedder:          Embeds ``content`` for upsert and queries for search.
        app_id:            The app partition id, stamped onto records for
                           self-containment / audit (the enforcement is the
                           scoped store layout, not this field).
        structured_collection / vector_collection: Collection names within the
                           scoped stores.
        max_retries:       Retries on an unparseable/invalid reconcile response.
        reconcile_guidance: Optional application-specific guidance injected as an
                           additive domain block in the reconcile prompt (see
                           :func:`_build_reconcile_prompt`).  The consolidation-
                           side analog of distill's ``domain_fact_guidance``;
                           ``None`` uses the generic prompt.
    """

    def __init__(
        self,
        structured_store: StructuredStoreBase,
        vector_store: VectorStoreBase,
        llm: LLMBase,
        embedder: EmbeddingBase,
        *,
        app_id: str | None = None,
        structured_collection: str = DEFAULT_STRUCTURED_COLLECTION,
        vector_collection: str = DEFAULT_VECTOR_COLLECTION,
        max_retries: int = 2,
        recall_neighbors: int = 5,
        reconcile_guidance: str | None = None,
    ) -> None:
        self._structured = structured_store
        self._vector = vector_store
        self._llm = llm
        self._embedder = embedder
        self._app_id = app_id
        self._structured_collection = structured_collection
        self._vector_collection = vector_collection
        self._recall_neighbors = recall_neighbors
        self._reconcile_prompt = _build_reconcile_prompt(reconcile_guidance)
        # Learned lazily: from the embedder at ``setup`` or the first real
        # embedding, then cached so ``_ensure`` only creates collections once.
        self._dimensions: int | None = None
        self._max_retries = max_retries
        self._ensured = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Eagerly create the structured + vector collections.  Idempotent.

        Optional: every operation lazily ensures its collections (the vector
        collection's dimensionality is learned from the first embedding), so the
        service is usable without calling this.  ``setup`` is for callers that
        want the collections to exist up front — its vector dimensionality is
        taken from the embedder (``EmbeddingBase.dimensions``).  When the embedder
        can't report it without an embedding call, eager creation is skipped and
        the collections are created on the first operation instead.
        """
        dims = self._embedder.dimensions
        if dims is None:
            return
        await self._ensure(dims)

    async def _ensure(self, dimensions: int) -> None:
        """Create both collections on first use; idempotent and cached.

        ``create_collection`` is idempotent across backends (safe on every
        startup), so this also re-registers the structured schema on a cold
        process — the registry ``save`` / ``query`` need without separate DDL.
        """
        if self._ensured:
            return
        self._dimensions = dimensions
        await self._structured.create_collection(
            LongTermRecord.collection_schema(self._structured_collection)
        )
        await self._vector.create_collection(
            VectorCollectionSchema(
                name=self._vector_collection,
                dimensions=dimensions,
                description=(
                    "Long-term memory content: durable facts, preferences, "
                    "corrections, and retrieval hints distilled from "
                    "session logs."
                ),
                metadata_fields=list(LongTermRecord.VECTOR_METADATA_FIELDS),
            )
        )
        self._ensured = True
        logger.info(
            "[long_term] app=%s collections ready: structured=%s vector=%s (dims=%d)",
            self._app_id, self._structured_collection,
            self._vector_collection, dimensions,
        )

    # ------------------------------------------------------------------
    # Recall (online, query-time)
    # ------------------------------------------------------------------

    async def recall(
        self, *, query: str, limit: int = 10
    ) -> list[LongTermRecord]:
        """Return active memories relevant to *query*, plus their neighborhood.

        The app partition is enforced by the scoped store, not here.  The first
        ``limit`` results are the vector-relevance hits; up to ``recall_neighbors``
        more are appended by following the memory graph one hop out from those
        hits (see :meth:`_neighbors`) — linked context the vector query alone
        would miss (a follow-up event, the other side of a contradiction).  Set
        ``recall_neighbors=0`` to disable and return only the relevance hits.
        """
        if not query.strip() or limit <= 0:
            return []
        # The vector index is active-only (see ``_save_record``), so every hit is
        # already active and ``limit`` results suffice — there is no status filter
        # to refill, so no over-fetch.
        hits = await self._search_content(query, top_k=limit)
        ordered_ids: list[str] = []
        for chunk in hits:
            if chunk.doc_id not in ordered_ids:
                ordered_ids.append(chunk.doc_id)
            if len(ordered_ids) >= limit:
                break
        if not ordered_ids:
            return []
        records = await self._load_records(ordered_ids)
        by_id = {r.memory_id: r for r in records}
        # Preserve vector-relevance order; drop any record that no longer exists.
        results = [by_id[mid] for mid in ordered_ids if mid in by_id]
        neighbors = await self._neighbors(results, self._recall_neighbors)
        logger.info(
            "[long_term] app=%s recall: query=%r limit=%d -> %d active record(s) "
            "+ %d neighbor(s)",
            self._app_id, query, limit, len(results), len(neighbors),
        )
        return results + neighbors

    async def _neighbors(
        self, primary: list[LongTermRecord], budget: int
    ) -> list[LongTermRecord]:
        """Active records one hop out from *primary* along the memory graph.

        The edges are bidirectional in effect: ``forward`` follows each primary
        record's own ``linked_memory_ids``; ``reverse`` finds records that link
        *to* a primary one (an overlaps scan pushed to the store), so a freshly
        recalled older memory still surfaces the newer memories that point back at
        it.  Excludes the primary set itself, dedupes, and caps at *budget*.
        """
        if budget <= 0 or not primary:
            return []
        primary_ids = {r.memory_id for r in primary}
        forward_ids = [
            mid
            for r in primary
            for mid in r.linked_memory_ids
            if mid not in primary_ids
        ]
        reverse_rows = await self._structured.query(
            self._structured_collection,
            [
                Col("status") == MemoryStatus.ACTIVE.value,
                Col("linked_memory_ids").overlaps(sorted(primary_ids)),
            ],
        )
        neighbors: dict[str, LongTermRecord] = {}
        if forward_ids:
            for rec in await self._load_records(list(dict.fromkeys(forward_ids))):
                if rec.status is MemoryStatus.ACTIVE and rec.memory_id not in primary_ids:
                    neighbors[rec.memory_id] = rec
        for row in reverse_rows:
            rec = LongTermRecord.model_validate(row)
            if rec.memory_id not in primary_ids:
                neighbors.setdefault(rec.memory_id, rec)
        return list(neighbors.values())[:budget]

    # ------------------------------------------------------------------
    # Lookup (the pull path: the memory_lookup tool)
    # ------------------------------------------------------------------

    async def lookup(
        self,
        *,
        query: str | None = None,
        kind: MemoryKind | None = None,
        entities: list[str] | None = None,
        limit: int = 10,
    ) -> list[LongTermRecord]:
        """Return active memories matching the given criteria.

        The pull counterpart to :meth:`recall` (the per-turn push): callers — in
        practice the query runner's ``memory_lookup`` tool — combine an optional
        semantic *query* with optional ``kind`` / ``entities`` filters.  Entity
        matching is exact on the normalized form.  The ``status=active`` filter
        is enforced here, never by the caller.  With a query, results follow
        vector-relevance order; without one, most recently updated first.
        """
        if limit <= 0:
            return []
        wanted_entities = set(normalize_entities(entities or []))
        if query and query.strip():
            # Over-fetch: the kind/entity filters are applied in Python, so fetch
            # a wider band than ``limit`` to refill what they drop.  No status
            # filter — the index is active-only (see ``_save_record``).
            hits = await self._search_content(query, top_k=max(limit * 4, limit))
            ordered_ids: list[str] = []
            for chunk in hits:
                if kind and chunk.metadata.get("kind") != kind.value:
                    continue
                if wanted_entities and not (
                    wanted_entities & set(chunk.metadata.get("entities") or [])
                ):
                    continue
                if chunk.doc_id not in ordered_ids:
                    ordered_ids.append(chunk.doc_id)
                if len(ordered_ids) >= limit:
                    break
            if not ordered_ids:
                return []
            by_id = {r.memory_id: r for r in await self._load_records(ordered_ids)}
            results = [by_id[mid] for mid in ordered_ids if mid in by_id]
            logger.info(
                "[long_term] app=%s lookup: query=%r kind=%s entities=%s -> %d record(s)",
                self._app_id, query, kind.value if kind else None,
                sorted(wanted_entities) or None, len(results),
            )
            return results

        # No query: a structured scan filtered by kind/status/entity overlap,
        # all pushed down to the store.
        filters = [Col("status") == MemoryStatus.ACTIVE.value]
        if kind:
            filters.append(Col("kind") == kind.value)
        if wanted_entities:
            filters.append(Col("entities").overlaps(sorted(wanted_entities)))
        rows = await self._structured.query(self._structured_collection, filters)
        records = [LongTermRecord.model_validate(row) for row in rows]
        records.sort(key=lambda r: r.updated_at, reverse=True)
        logger.info(
            "[long_term] app=%s lookup: kind=%s entities=%s -> %d record(s)",
            self._app_id, kind.value if kind else None,
            sorted(wanted_entities) or None, len(records[:limit]),
        )
        return records[:limit]

    # ------------------------------------------------------------------
    # Promotion review (the pending_review -> active gate)
    # ------------------------------------------------------------------

    async def list_pending(
        self,
        *,
        kind: MemoryKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[LongTermRecord]:
        """Return the gated records awaiting review, oldest first.

        The only reader of ``pending_review`` records: ``recall`` and ``lookup``
        are deliberately active-only, so a reviewer reaches gated facts/
        corrections here.  Ordered oldest-first so the queue is FIFO, and the
        full record is returned (``source_event_ids`` / ``evidence_snapshot``
        included) so the evidence can be audited before promotion.
        """
        if limit <= 0:
            return []
        filters = [Col("status") == MemoryStatus.PENDING_REVIEW.value]
        if kind:
            filters.append(Col("kind") == kind.value)
        rows = await self._structured.query(self._structured_collection, filters)
        records = [LongTermRecord.model_validate(row) for row in rows]
        records.sort(key=lambda r: r.created_at)
        page = records[offset : offset + limit]
        logger.info(
            "[long_term] app=%s list_pending: kind=%s -> %d of %d pending record(s)",
            self._app_id, kind.value if kind else None, len(page), len(records),
        )
        return page

    async def review_many(
        self, *, decisions: list[ReviewDecision]
    ) -> list[ReviewResult]:
        """Apply a batch of reviewer verdicts; return a per-item result.

        Accept promotes a record to ``active``; reject marks it ``superseded`` (a
        retraction marker consistent with reconcile's DELETE, keeping the audit
        trail rather than hard-deleting).  Idempotent: a record that is not
        currently ``pending_review`` is left untouched and reported ``skipped``,
        so a re-submitted decision can't resurrect a superseded record or
        re-promote one.

        Order-preserving and one result per decision.  All records are loaded in
        a single query and the survivors written in one batched pass (see
        :meth:`_save_records`) — the vector's ``status`` metadata flips alongside
        the structured row because ``recall`` filters on that metadata.  Decisions
        are evaluated in order against a shared in-memory view, so a duplicate id
        decides once and then reports ``skipped`` — the first flip is visible to
        the later occurrence.  Raises ``ValueError`` over :data:`MAX_REVIEW_BATCH`
        rather than silently truncating.
        """
        if len(decisions) > MAX_REVIEW_BATCH:
            raise ValueError(
                f"review batch of {len(decisions)} exceeds the maximum of {MAX_REVIEW_BATCH}"
            )
        if not decisions:
            return []
        records = await self._load_records(
            list(dict.fromkeys(d.memory_id for d in decisions))
        )
        by_id = {r.memory_id: r for r in records}
        results: list[ReviewResult] = []
        to_save: list[LongTermRecord] = []
        for decision in decisions:
            record = by_id.get(decision.memory_id)
            if record is None:
                outcome = ReviewOutcome.NOT_FOUND
            elif record.status is not MemoryStatus.PENDING_REVIEW:
                outcome = ReviewOutcome.SKIPPED
            else:
                record.status = (
                    MemoryStatus.ACTIVE if decision.accept else MemoryStatus.SUPERSEDED
                )
                record.updated_at = _utcnow()
                to_save.append(record)
                outcome = (
                    ReviewOutcome.ACCEPTED if decision.accept else ReviewOutcome.REJECTED
                )
            results.append(
                ReviewResult(memory_id=decision.memory_id, outcome=outcome)
            )
        if to_save:
            await self._save_records(to_save)
        logger.info(
            "[long_term] app=%s review_many: %d decision(s) applied, %d written",
            self._app_id, len(results), len(to_save),
        )
        return results

    # ------------------------------------------------------------------
    # Promote (the ADD path)
    # ------------------------------------------------------------------

    async def embed_contents(
        self, candidates: list[MemoryCandidate]
    ) -> dict[str, list[float]]:
        """Batch-embed candidate contents once for reuse across reconcile.

        Distillation reconciles a whole session's candidates in one pass; each
        reconcile would otherwise embed the candidate ``content`` twice — once as
        the related-records search query, once on the promote write.  Embedding
        all distinct contents up front in a single call collapses those N×2
        round-trips into one.  Returns a ``content -> embedding`` map; pass it
        straight back to :meth:`reconcile` as its ``embeddings`` argument, where
        the embed helpers reuse it on a hit and embed normally on a miss.
        """
        texts = list(dict.fromkeys(c.content for c in candidates if c.content))
        if not texts:
            return {}
        vectors = await self._embedder.embed(texts)
        await self._ensure(len(vectors[0]))
        return dict(zip(texts, vectors))

    async def promote(
        self,
        *,
        candidate: MemoryCandidate,
        status: MemoryStatus | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> str:
        """Write *candidate* as a new active/pending record; returns its id.

        Embeds ``content`` and writes both the structured record and its content
        vector with the provenance snapshot populated.  ``status`` defaults to
        the kind+confidence promotion gate (:data:`AUTO_PROMOTE_CONFIDENCE`):
        preferences/hints auto-active, a strongly-supported fact auto-active, and
        weaker facts / all corrections held for review.  ``embeddings`` is an
        optional ``content -> vector`` cache (see :meth:`embed_contents`) reused
        for the content embedding.
        """
        record = LongTermRecord(
            app_id=self._app_id,
            kind=candidate.kind,
            content=candidate.content,
            entities=normalize_entities(candidate.entities),
            confidence=candidate.confidence,
            status=status or self._default_status(candidate.kind, candidate.confidence),
            linked_memory_ids=list(dict.fromkeys(candidate.linked_memory_ids)),
            source_event_ids=list(candidate.source_event_ids),
            evidence_snapshot=dict(candidate.evidence_snapshot),
        )
        await self._save_record(record, embeddings=embeddings)
        logger.info(
            "[long_term] app=%s promote memory_id=%s kind=%s status=%s confidence=%.2f",
            self._app_id, record.memory_id, record.kind.value,
            record.status.value, record.confidence,
        )
        return record.memory_id

    # ------------------------------------------------------------------
    # Reconcile (the crux)
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        *,
        candidate: MemoryCandidate,
        embeddings: dict[str, list[float]] | None = None,
    ) -> str:
        """Merge *candidate* against accumulated belief; return the affected id.

        Vector-recalls related records, asks the LLM for one
        ``ReconcileOp``, and applies it.  ADD with no related records skips the
        LLM entirely.  On any LLM/parse failure the candidate is conservatively
        ADDed rather than dropped — never silently lose a candidate.

        ``embeddings`` is an optional ``content -> vector`` cache (see
        :meth:`embed_contents`) reused for the candidate's search query and
        promote write, so distilling a session embeds each content once.

        Scaling note — single-call additive extract+reconcile
        -----------------------------------------------------
        The current design is two-phase: ``Distiller`` makes one LLM call to
        extract N candidate facts from a session, then ``distill`` calls
        :meth:`reconcile` per candidate, and each candidate with related
        records spends one more LLM call in :meth:`_decide`.  That is **N+1
        LLM calls per session** (worst case), which is fine at design-partner
        volume but grows linearly with facts-per-session.

        mem0's V3 pipeline collapses this into a *single* additive call:
        before extraction it vector-recalls the top-K existing memories, then
        one LLM call sees ``(new turn, existing memories)`` and emits the
        extracted facts **together with** their ADD / UPDATE / DELETE / NOOP
        op against those memories — extraction and reconciliation in one shot.
        Net effect: one call per session instead of N+1, and the model
        reconciles the whole batch holistically rather than one fact at a time.

        Trade-offs if we adopt it:
          - Cost/latency drop from O(N) to O(1) calls — the reason to do it.
          - Lose the per-candidate audit point: today each :meth:`_decide`
            yields one ``ReconcileOp`` + reasoning for one (candidate, target)
            pair, which is easy to log and review; the batched call mixes all
            decisions into one response.
          - Anti-hallucination matters more: when many existing memories are
            shown at once, mask their UUIDs as small integers in the prompt
            and resolve back afterwards so the LLM can't invent/typo an id.
          - The pending-review gate and confidence reinforcement still apply,
            but must be re-derived from a batched response shape.

        Recommendation: keep the controllable per-candidate path as default;
        introduce the single-call mode as an opt-in for high-volume apps where
        per-session LLM cost dominates.  Revisit when that cost actually bites.
        """
        related = await self._related_records(candidate, embeddings=embeddings)

        if not related:
            logger.info(
                "[long_term] app=%s reconcile: no related records for kind=%s; "
                "promoting as new", self._app_id, candidate.kind.value,
            )
            return await self.promote(candidate=candidate, embeddings=embeddings)

        decision = await self._decide(candidate, related)
        op = decision.op
        target = next((r for r in related if r.memory_id == decision.target_memory_id), None)
        logger.info(
            "[long_term] app=%s reconcile: kind=%s vs %d related -> %s (target=%s)",
            self._app_id, candidate.kind.value, len(related), op.value,
            target.memory_id if target else None,
        )

        if op is ReconcileOp.NOOP:
            logger.info(
                "[long_term] app=%s reconcile NOOP (%s)", self._app_id, decision.reasoning
            )
            return target.memory_id if target else related[0].memory_id

        if op is ReconcileOp.UPDATE and target is not None:
            return await self._apply_update(
                target, candidate, decision.revised_content, embeddings=embeddings
            )

        if op is ReconcileOp.DELETE and target is not None:
            return await self._apply_delete(target, candidate, embeddings=embeddings)

        # ADD, or UPDATE/DELETE that named no resolvable target.
        return await self.promote(candidate=candidate, embeddings=embeddings)

    # ------------------------------------------------------------------
    # Reconcile internals
    # ------------------------------------------------------------------

    async def _apply_update(
        self,
        target: LongTermRecord,
        candidate: MemoryCandidate,
        revised: str | None,
        *,
        embeddings: dict[str, list[float]] | None = None,
    ) -> str:
        """Reinforce *target*: bump confidence, merge provenance, optionally revise."""
        target.confidence = min(1.0, target.confidence + CONFIDENCE_REINFORCE)
        target.entities = normalize_entities(target.entities + candidate.entities)
        # Merge the candidate's edges in, minus a self-edge to the record it is
        # reinforcing (the extractor may link the duplicate back to its target).
        target.linked_memory_ids = [
            mid
            for mid in dict.fromkeys(target.linked_memory_ids + candidate.linked_memory_ids)
            if mid != target.memory_id
        ]
        target.source_event_ids = _merge_refs(
            target.source_event_ids, candidate.source_event_ids
        )
        if candidate.evidence_snapshot:
            target.evidence_snapshot = {**target.evidence_snapshot, **candidate.evidence_snapshot}
        target.updated_at = _utcnow()
        content_changed = bool(revised) and revised != target.content
        if content_changed:
            target.content = revised  # type: ignore[assignment]
        await self._save_record(target, embeddings=embeddings)
        logger.info(
            "[long_term] app=%s reconcile UPDATE memory_id=%s confidence=%.2f revised=%s",
            self._app_id, target.memory_id, target.confidence, content_changed,
        )
        return target.memory_id

    async def _apply_delete(
        self,
        target: LongTermRecord,
        candidate: MemoryCandidate,
        *,
        embeddings: dict[str, list[float]] | None = None,
    ) -> str:
        """Supersede *target* and promote *candidate* — but only if it outranks.

        Resolution is by precedence (docs/long-term-memory.md#pipeline): a
        confirmed correction outranks an inferred fact, ties break on confidence,
        and the candidate is the more recent observation.  When the candidate
        does not outrank the target, the contradiction is *not* applied — the
        existing belief stands and the candidate is dropped as the weaker claim.
        """
        if not _outranks(candidate.kind, candidate.confidence, target):
            logger.info(
                "[long_term] app=%s reconcile DELETE rejected: candidate does not "
                "outrank memory_id=%s; keeping existing belief",
                self._app_id, target.memory_id,
            )
            return target.memory_id
        # Asymmetric visibility: the supersede takes effect instantly, the
        # replacement does not.  The contradicted target flips to ``superseded``
        # now and drops straight out of recall, but ``promote`` lands the
        # candidate at ``pending_review`` for any correction
        # (AUTO_PROMOTE_CONFIDENCE[CORRECTION] = 1.01, unreachable), so it adds
        # nothing active until a reviewer accepts it.  Net for the common case
        # of a user correcting a belief: the old belief leaves recall the moment
        # the correction lands, but the corrected belief stays invisible until
        # review — recall briefly knows neither.
        target.status = MemoryStatus.SUPERSEDED
        target.updated_at = _utcnow()
        await self._save_record(target)
        new_id = await self.promote(candidate=candidate, embeddings=embeddings)
        logger.info(
            "[long_term] app=%s reconcile DELETE superseded=%s replaced_by=%s",
            self._app_id, target.memory_id, new_id,
        )
        return new_id

    class _Decision:
        __slots__ = ("op", "target_memory_id", "revised_content", "reasoning")

        def __init__(self, op, target_memory_id, revised_content, reasoning):
            self.op = op
            self.target_memory_id = target_memory_id
            self.revised_content = revised_content
            self.reasoning = reasoning

    async def _decide(
        self, candidate: MemoryCandidate, related: list[LongTermRecord]
    ) -> "LongTermMemory._Decision":
        """Ask the LLM which ``ReconcileOp`` applies; ADD on any failure.

        Real ``memory_id`` UUIDs are masked to small integer ids (0, 1, 2…) in
        the prompt and resolved back afterwards: an LLM asked to echo a 36-char
        UUID will eventually typo or invent one, which would silently break the
        ``target_memory_id`` → record lookup downstream.
        """
        # index -> real memory_id; the LLM only ever sees the index.
        id_by_index = {i: r.memory_id for i, r in enumerate(related)}
        related_block = "\n".join(
            f"- id={i} kind={r.kind.value} confidence={r.confidence:.2f} "
            f"entities={r.entities} content={r.content!r}"
            for i, r in enumerate(related)
        )
        user = (
            f"NEW observation (kind={candidate.kind.value}):\n{candidate.content}\n\n"
            f"EXISTING related memories:\n{related_block}"
        )
        messages = [
            {"role": "system", "content": self._reconcile_prompt},
            {"role": "user", "content": user},
        ]
        parsed = await self._complete_json(messages)
        if parsed is None:
            logger.warning(
                "[long_term] app=%s reconcile decision unparseable; defaulting to ADD",
                self._app_id,
            )
            return self._Decision(ReconcileOp.ADD, None, None, "fallback: unparseable")
        try:
            op = ReconcileOp(parsed["operation"])
        except (KeyError, ValueError):
            return self._Decision(ReconcileOp.ADD, None, None, "fallback: bad operation")
        return self._Decision(
            op,
            self._resolve_target(parsed.get("target_memory_id"), id_by_index, op),
            parsed.get("revised_content"),
            parsed.get("reasoning", ""),
        )

    def _resolve_target(
        self, raw: object, id_by_index: dict[int, str], op: ReconcileOp
    ) -> str | None:
        """Map a masked integer id back to its real ``memory_id``.

        Returns ``None`` for anything that isn't a valid index, so an
        out-of-range or hallucinated value degrades to "no target" rather than
        pointing at the wrong record.  A non-null id that fails to resolve is
        logged: it means the LLM named an id we never showed it (so an
        UPDATE/DELETE silently degrades to ADD), which is worth seeing in
        telemetry rather than masking as a clean decision.
        """
        if raw is None:
            return None
        index: int | None
        try:
            index = int(raw)
        except (TypeError, ValueError):
            index = None
        resolved = id_by_index.get(index) if index is not None else None
        if resolved is None:
            logger.info(
                "[long_term] app=%s reconcile %s named unresolvable target id %r "
                "(valid 0..%d); degrading to ADD",
                self._app_id, op.value, raw, len(id_by_index) - 1,
            )
        return resolved

    async def _complete_json(self, messages: list[dict]) -> dict | None:
        """LLM call → parsed+validated JSON, retrying on failure (extraction pattern)."""
        for attempt in range(self._max_retries + 1):
            try:
                result = await self._llm.complete(messages)
                content = (result.get("content") or "").strip()
                parsed = json.loads(content)
                jsonschema.validate(instance=parsed, schema=_RECONCILE_SCHEMA)
                return parsed
            except (json.JSONDecodeError, jsonschema.ValidationError):
                if attempt < self._max_retries:
                    continue
                logger.warning(
                    "[long_term] app=%s reconcile JSON invalid after retries",
                    self._app_id, exc_info=True,
                )
                return None
            except Exception:
                logger.warning(
                    "[long_term] app=%s reconcile LLM call failed",
                    self._app_id, exc_info=True,
                )
                return None
        return None

    async def _related_records(
        self,
        candidate: MemoryCandidate,
        *,
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[LongTermRecord]:
        """Active records related to the candidate: vector ∪ entity overlap.

        Vector similarity alone misses paraphrased claims about the same entity
        (the contradiction then lands as a duplicate ADD), so records sharing a
        normalized entity are unioned in after the vector hits.
        """
        hits = await self._search_content(
            candidate.content, top_k=_RECONCILE_CANDIDATES * 3, embeddings=embeddings
        )
        ids = [
            c.doc_id
            for c in hits
            if c.metadata.get("status") == MemoryStatus.ACTIVE.value
        ][:_RECONCILE_CANDIDATES]
        for record in await self._entity_overlap_records(candidate.entities):
            if record.memory_id not in ids:
                ids.append(record.memory_id)
        if not ids:
            return []
        return await self._load_records(ids[:_RECONCILE_CANDIDATES * 2])

    async def _entity_overlap_records(
        self, entities: list[str]
    ) -> list[LongTermRecord]:
        """Active records sharing at least one normalized entity."""
        wanted = set(normalize_entities(entities))
        if not wanted:
            return []
        rows = await self._structured.query(
            self._structured_collection,
            [
                Col("status") == MemoryStatus.ACTIVE.value,
                Col("entities").overlaps(sorted(wanted)),
            ],
        )
        return [LongTermRecord.model_validate(row) for row in rows]

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    async def _save_record(
        self,
        record: LongTermRecord,
        *,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        """Upsert the structured row; keep the vector index active-only.

        The vector index holds only ``active`` records, because every vector
        reader (``recall``, ``lookup``, ``reconcile``'s ``_related_records``) is
        active-only.  Two statuses therefore stay out of it:

        - ``pending_review``: never indexed.  A gated record's vector is never
          read, and ``review`` embeds and upserts it on promotion to active — so
          writing it here is pure waste that gets overwritten.
        - ``superseded``: deleted from the index.  A record that *was* active and
          is now superseded must leave the vector store, or recall/reconcile keep
          matching it.  ``delete`` is a no-op when the record was never indexed
          (e.g. a rejected ``pending_review`` record), so it is safe for both
          paths into ``superseded``.

        ``content`` is still embedded on every path — not to be stored, but
        because the lazy ``_ensure`` learns the collection dimension from it (a
        cold process may not have created the structured collection yet, and the
        row below needs it).  The embed reuses a precomputed vector from
        ``embeddings`` when supplied, so on the distillation path it is a cache
        hit, not an API call.
        """
        embedding = await self._embed_text(record.content, embeddings)
        await self._ensure(len(embedding))
        await self._structured.save(self._structured_collection, [self._to_row(record)])
        if record.status is MemoryStatus.PENDING_REVIEW:
            return
        if record.status is MemoryStatus.SUPERSEDED:
            # Superseded leaves the vector index but the structured row stays.
            # TODO(retention): nothing GCs the structured store.  Superseded and
            # pending_review rows accumulate forever — long-term only ever *marks*
            # superseded (no hard delete) and nothing sets an expires_at, so the
            # collection grows monotonically.  Unlike episodic, which can drop a
            # whole session log, there is no retention sweep here.  Planned, not
            # yet built — a periodic sweep should age out superseded rows (and
            # stale pending_review) past a retention window.
            await self._vector.delete(self._vector_collection, [record.memory_id])
            return
        await self._vector.upsert(
            self._vector_collection,
            [
                Chunk(
                    chunk_id=record.memory_id,
                    doc_id=record.memory_id,
                    text=record.content,
                    embedding=embedding,
                    metadata=record.vector_metadata(),
                )
            ],
        )

    async def _save_records(self, records: list[LongTermRecord]) -> None:
        """Batch-write structured rows and reconcile the vector index for *records*.

        The batched mirror of :meth:`_save_record` for the review path, where a
        whole gated batch flips ``pending_review`` to ``active`` (accept) or
        ``superseded`` (reject) in one pass.  Following the same active-only
        invariant: accepted records are embedded and upserted into the index;
        rejected (``superseded``) records are deleted from it — a no-op here, since
        a gated record was never indexed, but kept for consistency with the
        active-only rule.  ``content`` is embedded in a single batch call (to
        learn the collection dimension and to feed the accepted upserts); one
        batch embed beats N round-trips.
        """
        if not records:
            return
        embeddings = await self._embedder.embed([r.content for r in records])
        await self._ensure(len(embeddings[0]))
        await self._structured.save(
            self._structured_collection, [self._to_row(r) for r in records]
        )
        chunks: list[Chunk] = []
        superseded_ids: list[str] = []
        for record, embedding in zip(records, embeddings):
            if record.status is MemoryStatus.SUPERSEDED:
                superseded_ids.append(record.memory_id)
                continue
            if record.status is MemoryStatus.PENDING_REVIEW:
                continue
            chunks.append(
                Chunk(
                    chunk_id=record.memory_id,
                    doc_id=record.memory_id,
                    text=record.content,
                    embedding=embedding,
                    metadata=record.vector_metadata(),
                )
            )
        if superseded_ids:
            await self._vector.delete(self._vector_collection, superseded_ids)
        if chunks:
            await self._vector.upsert(self._vector_collection, chunks)

    async def _search_content(
        self,
        query: str,
        *,
        top_k: int,
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[Chunk]:
        embedding = await self._embed_text(query, embeddings)
        await self._ensure(len(embedding))
        return await self._vector.search(
            self._vector_collection, query, embedding, top_k
        )

    async def _embed_text(
        self, text: str, cache: dict[str, list[float]] | None
    ) -> list[float]:
        """Embed *text*, reusing *cache* on a hit and embedding on a miss."""
        if cache is not None:
            cached = cache.get(text)
            if cached is not None:
                return cached
        return (await self._embedder.embed([text]))[0]

    async def _load_records(self, memory_ids: list[str]) -> list[LongTermRecord]:
        rows = await self._structured.query(
            self._structured_collection, [Col("memory_id").in_(memory_ids)]
        )
        return [LongTermRecord.model_validate(row) for row in rows]

    @staticmethod
    def _to_row(record: LongTermRecord) -> dict:
        """JSON-serializable row: ISO timestamps, enum values, refs as dicts."""
        return record.model_dump(mode="json")

    @staticmethod
    def _default_status(kind: MemoryKind, confidence: float) -> MemoryStatus:
        """Auto-active iff the score clears the kind's auto-promote threshold."""
        threshold = AUTO_PROMOTE_CONFIDENCE.get(kind, _FALLBACK_AUTO_PROMOTE)
        return (
            MemoryStatus.ACTIVE
            if confidence >= threshold
            else MemoryStatus.PENDING_REVIEW
        )


def _merge_refs(existing: list[EventRef], incoming: list[EventRef]) -> list[EventRef]:
    """Append new event refs, deduped by their identity triplet, order-preserving."""
    seen = {(r.session_id, r.seq, r.ulid) for r in existing}
    merged = list(existing)
    for ref in incoming:
        key = (ref.session_id, ref.seq, ref.ulid)
        if key not in seen:
            seen.add(key)
            merged.append(ref)
    return merged


def _outranks(kind: MemoryKind, confidence: float, target: LongTermRecord) -> bool:
    """Whether a candidate of (*kind*, *confidence*) supersedes *target*.

    A confirmed correction outranks any non-correction; otherwise the higher
    confidence wins, and an exact tie is resolved in the candidate's favour as
    the more recent observation.
    """
    cand_correction = kind is MemoryKind.CORRECTION
    target_correction = target.kind is MemoryKind.CORRECTION
    if cand_correction != target_correction:
        return cand_correction
    return confidence >= target.confidence
