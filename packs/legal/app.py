"""Legal contract analyst application — pre-configured ingestion + query stack.

``LegalContractApp`` bundles the full CogBase pipeline for legal contract
analysis into a single object, wiring together:

- ``ContractExtractor`` — extracts a structured summary from contract text via LLM
- ``Application``       — orchestrates chunking, embedding, and extraction
- ``Engine``            — routes queries and generates grounded answers

The vector store, embedder, and chunker are optional.  When omitted the app
operates in *structured-only* mode: contracts are processed for structured
extraction but raw text is not stored for semantic search (Pattern B queries
return empty results; Patterns A, C, D still work).

Typical usage (full mode)::

    import openai
    from packs.legal import LegalContractApp
    from cogbase.core.models import Document
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

    # Ingest a batch of contracts
    results = await app.ingest_many([
        Document(doc_id="vendor-001", text=vendor_contract),
        Document(doc_id="nda-002",    text=nda_text),
        Document(doc_id="lease-003",  text=lease_text),
    ])
    for r in results:
        status = f"{r.records_extracted} record extracted" if r.success else str(r.error)
        print(f"{r.doc_id}: {status}")

    result = await app.query("which contracts expire before 2026-01-01?")
    print(result.answer)

Structured-only mode (no vector search)::

    app = LegalContractApp(
        client=client,
        model="claude-sonnet-4-6",
        structured_store=SQLiteStructuredStore("contracts.db"),
    )
    await app.setup()
    await app.ingest(contract_text, doc_id="contract-001")
    result = await app.query("list all contracts with Acme Corp")
"""

from __future__ import annotations

from typing import Any, Sequence

from cogbase.core.application import Application, IngestResult, StructuredCollection, VectorCollection
from cogbase.core.models import Document
from cogbase.engine.engine import Engine
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.generation.llm import LLMGenerator
from cogbase.engine.retrieval.hybrid import HybridRetriever
from cogbase.engine.router import LLMRouter, QueryPattern
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.schema import CollectionSchema
from packs.legal.extractor import ContractExtractor
from packs.legal.schema import CONTRACTS_SCHEMA

# Patterns available when no vector store is configured (B and C require a vector store).
_STRUCTURED_ONLY_PATTERNS = [QueryPattern.A, QueryPattern.D]


# ---------------------------------------------------------------------------
# LegalContractApp
# ---------------------------------------------------------------------------


class LegalContractApp:
    """Pre-configured CogBase application for legal contract analysis.

    Bundles contract extraction, structured storage, and the full query engine
    under a small interface: ``setup`` → ``ingest`` / ``ingest_many`` → ``query``.

    Args:
        client:               Async OpenAI-compatible client used for both the
                              ``ContractExtractor`` and the query ``Engine``.
        model:                Model name forwarded to the extractor, router, and
                              generator (e.g. ``"claude-sonnet-4-6"``).
        structured_store:     Persistent store for extracted contract records.
        vector_store:         Vector store for raw contract text chunks.  Must be
                              provided together with *embedder* and *chunker*.
                              When ``None`` the app runs in structured-only mode.
        embedder:             Embedder for chunked contract text.  Required when
                              *vector_store* is supplied.
        chunker:              Chunker for splitting contract text.  Required when
                              *vector_store* is supplied.
        name:                 Logical name for the application.
        extractor_max_tokens: Max tokens for the ``ContractExtractor`` LLM call.
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

        extractor = ContractExtractor(client, model, max_tokens=extractor_max_tokens)

        structured_collections = [
            StructuredCollection(
                schema=CONTRACTS_SCHEMA,
                store=structured_store,
                extractor=extractor,
            )
        ]

        vector_collections: list[VectorCollection] = []
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
                available_patterns=None if vector_store else _STRUCTURED_ONLY_PATTERNS,
            ),
            retriever=HybridRetriever(
                structured_store=structured_store,
                vector_store=vector_store,
                embedder=embedder,
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
        """Ingest a single contract document.

        Chunks and embeds the text into the vector store (if configured) and
        extracts a structured ``ContractRecord`` into the structured store.

        Args:
            text:   Full contract text.
            doc_id: Stable identifier for the source document.
        """
        await self._app.ingest(text, doc_id)

    async def ingest_many(
        self,
        contracts: Sequence[Document | tuple[str, str]],
        *,
        concurrency: int = 5,
    ) -> list[IngestResult]:
        """Ingest a list of contracts, running up to *concurrency* at a time.

        Each contract is processed independently.  A failure on one document does
        not abort the others — the error is captured in the corresponding
        ``IngestResult`` and ingestion continues for the remaining documents.

        Results are returned in the same order as *contracts*.

        Args:
            contracts:   Sequence of ``Document`` objects **or**
                         ``(text, doc_id)`` tuples (both forms are accepted).
            concurrency: Maximum number of documents ingested simultaneously.
                         Defaults to ``5`` — a safe limit for LLM API rate caps.
                         Set to ``1`` for strictly sequential ingestion.

        Returns:
            ``list[IngestResult]`` in input order, one entry per document.
            Each result carries: ``doc_id``, ``success``, ``records_extracted``
            (0 or 1 — always 1 for a successfully parsed contract), and ``error``
            (the exception raised, or ``None`` on success).

        Example::

            results = await app.ingest_many([
                Document(doc_id="vendor-001", text=vendor_text),
                Document(doc_id="nda-002",    text=nda_text),
            ])
            ok     = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
        """
        return await self._app.ingest_many(contracts, concurrency=concurrency)

    async def query(self, text: str) -> GenerationResult:
        """Answer a natural-language query over ingested contracts.

        Automatically routes to the correct retrieval pattern:

        - Pattern A — structured lookup (no LLM call needed)
        - Pattern B — semantic search over raw contract text
        - Pattern C — hybrid reasoning across structured records and text
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
