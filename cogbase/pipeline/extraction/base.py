"""Abstract contract for document extractors."""

import abc
import asyncio

from pydantic import BaseModel

from cogbase.core.models import Document
from cogbase.stores import CollectionSchema


class ExtractorBase(abc.ABC):
    """Extract structured records from document text and declare where to store them.

    An extractor is responsible for three things:
    1. Declaring the **collection name** it writes to (``collection`` property).
    2. Declaring the **schema** of that collection (``schema`` property), so the
       pipeline can call ``create_collection`` idempotently on startup.
    3. Implementing ``_extract_once`` to turn raw text into a single Pydantic record.

    The public ``extract`` method wraps ``_extract_once`` with automatic retries
    and exponential backoff, so transient LLM JSON failures are handled for all
    extractors without any per-extractor boilerplate.  A ``None`` return from
    ``_extract_once`` is treated as a parse failure and triggers a retry; an empty
    list is treated as a valid (no-data) result and is returned immediately.

    The output type is intentionally open — any ``BaseModel`` subclass works:
    ``Fact``, ``Entity``, ``Clause``, ``Event``, ``Relationship``, etc.  Domain
    packs define their own model + extractor pairs and register them in the
    pipeline without touching core CogBase code.

    Args:
        max_retries: Number of additional attempts after the first failure.
                     Sleep between attempts is ``2^(attempt-1)`` seconds
                     (1 s, 2 s, 4 s, …).  Default: 2.

    Example::

        class ClauseExtractor(ExtractorBase):
            @property
            def collection(self) -> str:
                return "clauses"

            @property
            def schema(self) -> CollectionSchema:
                return CollectionSchema(
                    name="clauses",
                    description="Extracted contract clauses with type, text, and page reference.",
                    primary_fields=["clause_id"],
                    fields={
                        "clause_id": FieldSchema(type=FieldType.STRING),
                        "doc_id":    FieldSchema(type=FieldType.STRING, index=True),
                        "type":      FieldSchema(type=FieldType.STRING, index=True),
                        "text":      FieldSchema(type=FieldType.STRING),
                        "page":      FieldSchema(type=FieldType.INTEGER, nullable=True),
                    },
                )

            async def _extract_once(self, doc: Document) -> BaseModel | None:
                ...
    """

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries

    @property
    @abc.abstractmethod
    def collection(self) -> str:
        """Name of the structured store collection this extractor writes to."""

    @property
    @abc.abstractmethod
    def schema(self) -> CollectionSchema:
        """Schema for ``collection``.

        The pipeline calls ``structured_store.create_collection(extractor.schema)``
        before the first ``save``.  ``create_collection`` is idempotent, so this is
        safe to call on every pipeline run.
        """

    @abc.abstractmethod
    async def _extract_once(self, doc: Document) -> BaseModel | None:
        """Single extraction attempt for *doc*.

        Called by ``extract``; do not call directly.  Implement the LLM call (or
        any other extraction logic) here and return a Pydantic record on success or
        ``None`` on parse failure.  Return ``None`` only when the output is
        unparseable — not when the document simply contains no matching data (return
        an appropriate empty/default record or an empty list instead).

        Args:
            doc: Source document whose ``text`` is passed to the extractor and
                 whose ``doc_id`` should be propagated onto the returned record.

        Returns:
            A single Pydantic record whose fields match ``self.schema``, or
            ``None`` when the extractor cannot produce a valid result.
        """

    async def extract(self, doc: Document) -> BaseModel | None:
        """Extract a record from *doc*, retrying on parse failures.

        Returns ``None`` immediately for blank ``doc.text``.  Otherwise calls
        ``_extract_once`` up to ``max_retries + 1`` times, sleeping
        ``2^(attempt-1)`` seconds between attempts.  Returns the first non-None
        result, or ``None`` after all attempts are exhausted.
        """
        if not doc.text.strip():
            return None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                await asyncio.sleep(2 ** (attempt - 1))
            result = await self._extract_once(doc)
            if result is not None:
                return result

        return None
