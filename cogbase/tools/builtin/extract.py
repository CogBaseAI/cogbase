"""Built-in tool: extract structured records from a document and save to a structured store."""

import logging

from cogbase.core.models import Document
from cogbase.core.session import Session
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores.base import StructuredStoreBase
from cogbase.tools.base import Tool

logger = logging.getLogger(__name__)


class ExtractTool(Tool):
    """Extract a structured record from a document and save it to a structured store.

    Input dict keys:
    - ``document`` (Document): the document to process.

    Output dict keys:
    - ``doc_id`` (str): identifier of the processed document.
    - ``extracted`` (bool): ``True`` when a record was successfully extracted and saved.
    """

    name = "extract-structured"
    description = (
        "Run a domain-specific extractor over a document to pull out typed, structured "
        "records (facts, clauses, entities, etc.) and persist them to the configured "
        "structured store collection. Returns whether a record was extracted."
    )

    def __init__(
        self,
        extractor: ExtractorBase,
        structured_store: StructuredStoreBase,
    ) -> None:
        self._extractor = extractor
        self._structured_store = structured_store

    async def run(self, input: dict, session: Session) -> dict:
        """Extract a structured record from *input["document"]* and save it.

        Args:
            input:   Must contain ``"document"`` (a ``Document`` instance).
            session: Active session for log correlation.

        Returns:
            ``{"doc_id": str, "extracted": bool}``

        Raises:
            KeyError: If ``"document"`` is missing from *input*.
            TypeError: If ``input["document"]`` is not a ``Document``.
        """
        doc: Document = input["document"]
        if not isinstance(doc, Document):
            raise TypeError(f"input['document'] must be a Document, got {type(doc)}")

        logger.info(
            "extract-structured.start session=%s doc_id=%s collection=%s",
            session.session_id,
            doc.doc_id,
            self._extractor.collection,
        )

        record = await self._extractor.extract(doc)
        if record is None:
            logger.debug(
                "extract-structured.no-record session=%s doc_id=%s",
                session.session_id,
                doc.doc_id,
            )
            return {"doc_id": doc.doc_id, "extracted": False}

        await self._structured_store.save(self._extractor.collection, [record])

        logger.info(
            "extract-structured.done session=%s doc_id=%s collection=%s",
            session.session_id,
            doc.doc_id,
            self._extractor.collection,
        )
        return {"doc_id": doc.doc_id, "extracted": True}
