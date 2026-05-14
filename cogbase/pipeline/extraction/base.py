"""Abstract contract for document extractors."""

import abc
import asyncio
import logging

from pydantic import BaseModel

from cogbase.core.models import Document

logger = logging.getLogger(__name__)

class ExtractorBase(abc.ABC):
    """Extract structured records from document text.

    Implements ``_extract_once`` to turn raw document text into Pydantic records.
    The public ``extract`` method adds automatic retries with exponential backoff;
    a ``None`` return from ``_extract_once`` triggers a retry, an empty list is
    returned immediately.

    The output type is intentionally open — any ``BaseModel`` subclass works.

    Args:
        max_retries: Number of additional attempts after the first failure.
                     Sleep between attempts is ``2^(attempt-1)`` seconds
                     (1 s, 2 s, 4 s, …).  Default: 2.

    Example::

        class ClauseExtractor(ExtractorBase):
            async def _extract_once(self, doc: Document) -> list[BaseModel] | None:
                ...
    """

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries

    @abc.abstractmethod
    async def _extract_once(self, doc: Document) -> list[BaseModel] | None:
        """Single extraction attempt for *doc*.

        Called by ``extract``; do not call directly.  Return a list of Pydantic
        records on success, an empty list when the document contains no matching
        data, or ``None`` on parse failure (triggers a retry).

        Args:
            doc: Source document whose ``text`` is passed to the extractor and
                 whose ``doc_id`` should be propagated onto the returned records.

        Returns:
            A list of Pydantic records whose fields match ``self.schema``,
            an empty list when no data is found, or ``None`` on parse failure.
        """

    async def extract(self, doc: Document) -> list[BaseModel] | None:
        """Extract records from *doc*, retrying on parse failures.

        Returns ``None`` immediately for blank ``doc.text``.  Otherwise calls
        ``_extract_once`` up to ``max_retries + 1`` times, sleeping
        ``2^(attempt-1)`` seconds between attempts.  Returns the first non-None
        result (including an empty list), or ``None`` after all attempts are
        exhausted.
        """
        if not doc.text.strip():
            return None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                await asyncio.sleep(2 ** (attempt - 1))
            result = await self._extract_once(doc)
            if result is not None:
                return result

        logger.error("failed to extract, doc_id=%s", doc.doc_id)
        return None
