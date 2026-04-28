"""Built-in tool: extract structured records from a document and save to a structured store."""

import json
import logging

from cogbase.core.models import Document
from cogbase.llms.base import SystemTool, ToolDefinition
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores import StructuredStoreBase

logger = logging.getLogger(__name__)


class ExtractTool(SystemTool):
    """Extract a structured record from a document and save it to a structured store.

    Handler input keys:
    - ``document`` (Document): the document to process.

    Handler result keys (JSON-encoded string):
    - ``doc_id`` (str): identifier of the processed document.
    - ``extracted`` (bool): ``True`` when a record was successfully extracted and saved.
    """

    def __init__(
        self,
        extractor: ExtractorBase,
        structured_store: StructuredStoreBase,
    ) -> None:
        _extractor = extractor
        _structured_store = structured_store

        async def _handler(inputs: dict) -> str:
            doc: Document = inputs["document"]
            if not isinstance(doc, Document):
                raise TypeError(f"inputs['document'] must be a Document, got {type(doc)}")

            logger.info(
                "extract-structured.start doc_id=%s collection=%s",
                doc.doc_id,
                _extractor.collection,
            )

            record = await _extractor.extract(doc)
            if record is None:
                logger.debug("extract-structured.no-record doc_id=%s", doc.doc_id)
                return json.dumps({"doc_id": doc.doc_id, "extracted": False})

            await _structured_store.save(_extractor.collection, [record])

            logger.info(
                "extract-structured.done doc_id=%s collection=%s",
                doc.doc_id,
                _extractor.collection,
            )
            return json.dumps({"doc_id": doc.doc_id, "extracted": True})

        super().__init__(
            definition=ToolDefinition(
                name="extract-structured",
                description=(
                    "Run a domain-specific extractor over a document to pull out typed, structured "
                    "records (facts, clauses, entities, etc.) and persist them to the configured "
                    "structured store collection. Returns whether a record was extracted."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            handler=_handler,
        )
