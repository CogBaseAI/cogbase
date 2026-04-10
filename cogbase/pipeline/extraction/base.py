"""Abstract contract for document extractors."""

import abc

from pydantic import BaseModel

from cogbase.stores.schema import CollectionSchema


class ExtractorBase(abc.ABC):
    """Extract structured records from document text and declare where to store them.

    An extractor is responsible for three things:
    1. Declaring the **collection name** it writes to (``collection`` property).
    2. Declaring the **schema** of that collection (``schema`` property), so the
       pipeline can call ``create_collection`` idempotently on startup.
    3. Implementing ``extract`` to turn raw text into a list of Pydantic records.

    The output type is intentionally open — any ``BaseModel`` subclass works:
    ``Fact``, ``Entity``, ``Clause``, ``Event``, ``Relationship``, etc.  Domain
    packs define their own model + extractor pairs and register them in the
    pipeline without touching core CogBase code.

    Example::

        class ClauseExtractor(ExtractorBase):
            @property
            def collection(self) -> str:
                return "clauses"

            @property
            def schema(self) -> CollectionSchema:
                return CollectionSchema(
                    name="clauses",
                    id_field="clause_id",
                    fields={
                        "clause_id": FieldSchema(type=FieldType.STRING),
                        "doc_id":    FieldSchema(type=FieldType.STRING, index=True),
                        "type":      FieldSchema(type=FieldType.STRING, index=True),
                        "text":      FieldSchema(type=FieldType.STRING),
                        "page":      FieldSchema(type=FieldType.INTEGER, nullable=True),
                    },
                )

            async def extract(self, text: str, doc_id: str) -> list[BaseModel]:
                ...
    """

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
    async def extract(self, text: str, doc_id: str) -> list[BaseModel]:
        """Return extracted records for *text*.

        Args:
            text:   Full or chunked document text passed to this extractor.
            doc_id: Stable identifier of the source document; implementations
                    should propagate it onto every returned record.

        Returns:
            A (possibly empty) list of Pydantic records.  All records must be
            instances of the same ``BaseModel`` subclass — the one whose fields
            match ``self.schema``.
        """
