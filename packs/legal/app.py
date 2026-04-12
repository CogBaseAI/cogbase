"""Legal contract analyst application — pre-configured ingestion + query stack.

``LegalContractApp`` bundles the full CogBase pipeline for legal contract
analysis into a single object, wiring together:

- ``ClauseExtractor``  — extracts typed clauses from contract text via LLM
- ``Application``      — orchestrates chunking, embedding, and extraction
- ``Engine``           — routes queries and generates grounded answers

The vector store, embedder, and chunker are optional.  When omitted the app
operates in *structured-only* mode: contracts are processed for clause
extraction but raw text is not stored for semantic search (Pattern B queries
return empty results; Patterns A, C, D still work).

Typical usage (full mode)::

    import openai
    from packs.legal import LegalContractApp
    from cogbase.stores.structured.sqlite import SQLiteStructuredStore
    from cogbase.stores.vector.faiss_store import FAISSVectorStore
    from cogbase.pipeline.ingestion.embedder import SentenceTransformersEmbedder
    from cogbase.pipeline.ingestion.fixed import FixedSizeChunker

    client = openai.AsyncOpenAI(api_key="...")
    app = LegalContractApp(
        client=client,
        model="claude-sonnet-4-6",
        structured_store=SQLiteStructuredStore("contracts.db"),
        vector_store=FAISSVectorStore(dim=384),
        embedder=SentenceTransformersEmbedder(),
        chunker=FixedSizeChunker(chunk_size=512, overlap=64),
    )
    await app.setup()
    await app.ingest(contract_text, doc_id="contract-001")
    result = await app.query("what are the termination clauses?")
    print(result.answer)

Structured-only mode (no vector search)::

    app = LegalContractApp(
        client=client,
        model="claude-sonnet-4-6",
        structured_store=SQLiteStructuredStore("contracts.db"),
    )
    await app.setup()
    await app.ingest(contract_text, doc_id="contract-001")
    result = await app.query("list all payment clauses")   # routes Pattern A
"""

from __future__ import annotations

from typing import Any

from cogbase.core.application import Application, StructuredCollection, VectorCollection
from cogbase.core.models import Chunk
from cogbase.engine.engine import Engine
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.generation.llm import LLMGenerator
from cogbase.engine.retrieval.hybrid import HybridRetriever
from cogbase.engine.router import LLMRouter
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.schema import CollectionSchema
from packs.legal.extractor import ClauseExtractor
from packs.legal.schema import CLAUSES_SCHEMA


# ---------------------------------------------------------------------------
# Null-object adapters (used when vector params are absent)
# ---------------------------------------------------------------------------


class _NullVectorStore(VectorStoreBase):
    """No-op vector store — returns empty results, raises on search if called."""

    async def upsert(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        pass  # unreachable: no VectorCollection is added in structured-only mode

    async def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        return []

    async def delete(self, doc_id: str) -> None:  # pragma: no cover
        pass


class _NullEmbedder(EmbedderBase):
    """No-op embedder — attaches a zero-length dummy embedding so VectorRetriever
    can proceed; _NullVectorStore.search ignores the embedding and returns []."""

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        return [c.model_copy(update={"embedding": []}) for c in chunks]


# ---------------------------------------------------------------------------
# LegalContractApp
# ---------------------------------------------------------------------------


class LegalContractApp:
    """Pre-configured CogBase application for legal contract analysis.

    Bundles clause extraction, structured storage, and the full query engine
    under a single three-method interface: ``setup`` → ``ingest`` → ``query``.

    Args:
        client:               Async OpenAI-compatible client used for both the
                              ``ClauseExtractor`` and the query ``Engine``.
        model:                Model name forwarded to the extractor, router, and
                              generator (e.g. ``"claude-sonnet-4-6"``).
        structured_store:     Persistent store for extracted clauses.
        vector_store:         Vector store for raw contract text chunks.  Must be
                              provided together with *embedder* and *chunker*.
                              When ``None`` the app runs in structured-only mode.
        embedder:             Embedder for chunked contract text.  Required when
                              *vector_store* is supplied.
        chunker:              Chunker for splitting contract text.  Required when
                              *vector_store* is supplied.
        name:                 Logical name for the application.
        extractor_max_tokens: Max tokens for the ``ClauseExtractor`` LLM call.
        generator_max_tokens: Max tokens for the ``LLMGenerator`` LLM call.
        retriever_top_k:      Number of nearest-neighbour chunks to return from
                              the vector store on semantic queries.

    Raises:
        ValueError: If only some of *vector_store*, *embedder*, *chunker* are
                    supplied — all three must be present or all absent.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        structured_store: StructuredStoreBase,
        *,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
        chunker: ChunkerBase | None = None,
        name: str = "legal",
        extractor_max_tokens: int = 4096,
        generator_max_tokens: int = 1024,
        retriever_top_k: int = 10,
    ) -> None:
        vector_params = (vector_store, embedder, chunker)
        n_provided = sum(p is not None for p in vector_params)
        if 0 < n_provided < 3:
            raise ValueError(
                "vector_store, embedder, and chunker must all be provided together "
                "or all omitted. Received a partial set."
            )

        extractor = ClauseExtractor(client, model, max_tokens=extractor_max_tokens)

        structured_collections = [
            StructuredCollection(
                schema=CLAUSES_SCHEMA,
                store=structured_store,
                extractor=extractor,
            )
        ]

        vector_collections: list[VectorCollection] = []
        effective_vector_store: VectorStoreBase
        effective_embedder: EmbedderBase

        if vector_store is not None:
            assert embedder is not None and chunker is not None  # validated above
            vector_collections.append(
                VectorCollection(
                    name="documents",
                    store=vector_store,
                    embedder=embedder,
                    chunker=chunker,
                )
            )
            effective_vector_store = vector_store
            effective_embedder = embedder
        else:
            effective_vector_store = _NullVectorStore()
            effective_embedder = _NullEmbedder()

        self._app = Application(
            name=name,
            vector_collections=vector_collections,
            structured_collections=structured_collections,
        )

        self._engine = Engine(
            router=LLMRouter(
                client,
                model,
                schema=self._app.structured_schemas,
            ),
            retriever=HybridRetriever(
                structured_store=structured_store,
                vector_store=effective_vector_store,
                embedder=effective_embedder,
                top_k=retriever_top_k,
            ),
            generator=LLMGenerator(client, model, max_tokens=generator_max_tokens),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create all collections in their respective stores. Idempotent."""
        await self._app.setup()

    async def ingest(self, text: str, doc_id: str) -> None:
        """Ingest a contract document.

        Chunks and embeds the text into the vector store (if configured) and
        extracts typed clauses into the structured store.

        Args:
            text:   Full contract text.
            doc_id: Stable identifier for the source document.
        """
        await self._app.ingest(text, doc_id)

    async def query(self, text: str) -> GenerationResult:
        """Answer a natural-language query over ingested contracts.

        Automatically routes to the correct retrieval pattern:

        - Pattern A — structured clause lookup (no LLM call needed)
        - Pattern B — semantic search over raw contract text
        - Pattern C — hybrid reasoning across clauses and text
        - Pattern D — grounded report with ``[FINDINGS]`` / ``[SUPPORTING_QUOTES]``

        Args:
            text: Natural-language question or instruction.

        Returns:
            ``GenerationResult`` with at minimum an ``answer`` string.
            Pattern D results also populate ``findings`` and ``supporting_quotes``.
        """
        return await self._engine.query(text)

    # ------------------------------------------------------------------
    # Accessors (advanced use)
    # ------------------------------------------------------------------

    @property
    def application(self) -> Application:
        """The underlying ``Application`` (ingestion layer)."""
        return self._app

    @property
    def engine(self) -> Engine:
        """The underlying ``Engine`` (query layer)."""
        return self._engine

    @property
    def structured_schemas(self) -> list[CollectionSchema]:
        """Schemas for all structured collections (convenience proxy)."""
        return self._app.structured_schemas
