"""Built-in tool: chunk a document, embed the chunks, upsert into a vector store."""

import json
import logging

from cogbase.core.models import Document
from cogbase.embeddings.base import EmbeddingBase
from cogbase.llms.base import SystemTool, ToolDefinition
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.stores import VectorStoreBase

logger = logging.getLogger(__name__)


class ChunkEmbedUpsertTool(SystemTool):
    """Split a document into chunks, embed them, and upsert into a vector store.

    Handler input keys:
    - ``document`` (Document): the document to process.

    Handler result keys (JSON-encoded string):
    - ``doc_id`` (str): identifier of the processed document.
    - ``chunks_upserted`` (int): number of chunks written to the vector store.
    """

    def __init__(
        self,
        chunker: ChunkerBase,
        embedder: EmbeddingBase,
        vector_store: VectorStoreBase,
        collection_name: str,
    ) -> None:
        _chunker = chunker
        _embedder = embedder
        _vector_store = vector_store
        _collection_name = collection_name

        async def _handler(inputs: dict) -> str:
            doc: Document = inputs["document"]
            if not isinstance(doc, Document):
                raise TypeError(f"inputs['document'] must be a Document, got {type(doc)}")

            logger.info(
                "chunk-embed-upsert.start doc_id=%s collection=%s",
                doc.doc_id,
                _collection_name,
            )

            chunks = _chunker.chunk(doc)
            if not chunks:
                logger.debug("chunk-embed-upsert.no-chunks doc_id=%s", doc.doc_id)
                return json.dumps({"doc_id": doc.doc_id, "chunks_upserted": 0})

            embeddings = await _embedder.embed([c.text for c in chunks])
            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks."
                )

            embedded = [
                chunk.model_copy(update={"embedding": emb})
                for chunk, emb in zip(chunks, embeddings)
            ]
            await _vector_store.upsert(_collection_name, embedded)

            logger.info(
                "chunk-embed-upsert.done doc_id=%s collection=%s chunks=%d",
                doc.doc_id,
                _collection_name,
                len(embedded),
            )
            return json.dumps({"doc_id": doc.doc_id, "chunks_upserted": len(embedded)})

        super().__init__(
            definition=ToolDefinition(
                name="chunk-embed-upsert",
                description=(
                    "Split a document into chunks, generate dense embeddings for each chunk, "
                    "and upsert the embedded chunks into the configured vector store collection. "
                    "Returns the number of chunks written."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            handler=_handler,
        )
