"""Built-in tool: chunk a document, embed the chunks, upsert into a vector store."""

import logging

from cogbase.core.models import Document
from cogbase.core.session import Session
from cogbase.embeddings.base import EmbeddingBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.stores.base import VectorStoreBase
from cogbase.tools.base import Tool

logger = logging.getLogger(__name__)


class ChunkEmbedUpsertTool(Tool):
    """Split a document into chunks, embed them, and upsert into a vector store.

    Input dict keys:
    - ``document`` (Document): the document to process.

    Output dict keys:
    - ``doc_id`` (str): identifier of the processed document.
    - ``chunks_upserted`` (int): number of chunks written to the vector store.
      Zero when the document text is empty or produces no chunks.
    """

    name = "chunk-embed-upsert"
    description = (
        "Split a document into chunks, generate dense embeddings for each chunk, "
        "and upsert the embedded chunks into the configured vector store collection. "
        "Returns the number of chunks written."
    )

    def __init__(
        self,
        chunker: ChunkerBase,
        embedder: EmbeddingBase,
        vector_store: VectorStoreBase,
        collection_name: str,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._collection_name = collection_name

    async def run(self, input: dict, session: Session) -> dict:
        """Chunk, embed, and upsert the document in *input["document"]*.

        Args:
            input:   Must contain ``"document"`` (a ``Document`` instance).
            session: Active session for log correlation.

        Returns:
            ``{"doc_id": str, "chunks_upserted": int}``

        Raises:
            KeyError: If ``"document"`` is missing from *input*.
            TypeError: If ``input["document"]`` is not a ``Document``.
            ValueError: If the embedder returns a different number of vectors
                        than there are chunks.
        """
        doc: Document = input["document"]
        if not isinstance(doc, Document):
            raise TypeError(f"input['document'] must be a Document, got {type(doc)}")

        logger.info(
            "chunk-embed-upsert.start session=%s doc_id=%s collection=%s",
            session.session_id,
            doc.doc_id,
            self._collection_name,
        )

        chunks = self._chunker.chunk(doc)
        if not chunks:
            logger.debug(
                "chunk-embed-upsert.no-chunks session=%s doc_id=%s",
                session.session_id,
                doc.doc_id,
            )
            return {"doc_id": doc.doc_id, "chunks_upserted": 0}

        embeddings = await self._embedder.embed([c.text for c in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks."
            )

        embedded = [
            chunk.model_copy(update={"embedding": embedding})
            for chunk, embedding in zip(chunks, embeddings)
        ]
        await self._vector_store.upsert(self._collection_name, embedded)

        logger.info(
            "chunk-embed-upsert.done session=%s doc_id=%s collection=%s chunks=%d",
            session.session_id,
            doc.doc_id,
            self._collection_name,
            len(embedded),
        )
        return {"doc_id": doc.doc_id, "chunks_upserted": len(embedded)}
