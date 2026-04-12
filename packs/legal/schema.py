"""Schema and Pydantic model for the legal contract review pack."""

from __future__ import annotations

from pydantic import BaseModel

from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType

CLAUSES_COLLECTION = "clauses"

CLAUSES_SCHEMA = CollectionSchema(
    name=CLAUSES_COLLECTION,
    id_field="clause_id",
    fields={
        "clause_id":   FieldSchema(type=FieldType.STRING),
        "doc_id":      FieldSchema(type=FieldType.STRING, index=True),
        "type":        FieldSchema(type=FieldType.STRING, index=True),
        "text":        FieldSchema(type=FieldType.STRING),
        "page":        FieldSchema(type=FieldType.INTEGER, nullable=True),
        "confidence":  FieldSchema(type=FieldType.FLOAT),
    },
)


class Clause(BaseModel):
    """A single extracted contract clause.

    Attributes:
        clause_id:  Stable unique ID: ``{doc_id}_{type}_{index}``.
        doc_id:     Source document identifier.
        type:       Clause category. One of: ``payment``, ``termination``,
                    ``liability``, ``notice``, ``governing_law``, ``confidentiality``,
                    ``indemnification``, ``dispute_resolution``, ``other``.
        text:       Verbatim clause text as it appears in the contract.
        page:       Page number (1-indexed), if available.
        confidence: Extractor confidence in [0.0, 1.0].
    """

    clause_id: str
    doc_id: str
    type: str
    text: str
    page: int | None = None
    confidence: float
