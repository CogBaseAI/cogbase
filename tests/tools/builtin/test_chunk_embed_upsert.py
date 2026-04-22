"""Unit tests for ChunkEmbedUpsertTool."""

import pytest

from cogbase.core.models import Chunk, Document
from cogbase.core.session import Session
from cogbase.tools.builtin.chunk_embed_upsert import ChunkEmbedUpsertTool


def _doc(text: str = "Hello world.") -> Document:
    return Document(doc_id="doc-1", text=text)


def _session() -> Session:
    return Session()


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(chunk_id=f"doc-1_{idx}", doc_id="doc-1", text=text)


# ---------------------------------------------------------------------------
# Stub dependencies
# ---------------------------------------------------------------------------

class StubChunker:
    def __init__(self, chunks):
        self._chunks = chunks

    def chunk(self, doc: Document):
        return self._chunks


class StubEmbedder:
    def __init__(self, vectors):
        self._vectors = vectors

    async def embed(self, texts):
        return self._vectors[: len(texts)]


class StubEmbedderMismatch:
    """Returns one extra embedding to trigger the mismatch error."""

    async def embed(self, texts):
        return [[0.1]] * (len(texts) + 1)


class StubVectorStore:
    def __init__(self):
        self.upserted = []

    async def upsert(self, collection, chunks):
        self.upserted.extend(chunks)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_returns_correct_count():
    chunks = [_chunk("chunk 0", 0), _chunk("chunk 1", 1)]
    embedder = StubEmbedder([[0.1, 0.2], [0.3, 0.4]])
    store = StubVectorStore()
    tool = ChunkEmbedUpsertTool(
        chunker=StubChunker(chunks),
        embedder=embedder,
        vector_store=store,
        collection_name="col",
    )
    result = await tool.run({"document": _doc()}, _session())
    assert result == {"doc_id": "doc-1", "chunks_upserted": 2}


@pytest.mark.asyncio
async def test_run_upserts_embeddings_onto_chunks():
    chunk = _chunk("hello", 0)
    embedder = StubEmbedder([[1.0, 2.0]])
    store = StubVectorStore()
    tool = ChunkEmbedUpsertTool(
        chunker=StubChunker([chunk]),
        embedder=embedder,
        vector_store=store,
        collection_name="col",
    )
    await tool.run({"document": _doc()}, _session())
    assert store.upserted[0].embedding == [1.0, 2.0]


@pytest.mark.asyncio
async def test_run_uses_correct_collection():
    chunk = _chunk("x", 0)
    store = StubVectorStore()

    class TrackingStore:
        def __init__(self):
            self.collections = []
            self.upserted = []

        async def upsert(self, collection, chunks):
            self.collections.append(collection)
            self.upserted.extend(chunks)

    tracking = TrackingStore()
    tool = ChunkEmbedUpsertTool(
        chunker=StubChunker([chunk]),
        embedder=StubEmbedder([[0.0]]),
        vector_store=tracking,
        collection_name="my-col",
    )
    await tool.run({"document": _doc()}, _session())
    assert tracking.collections == ["my-col"]


# ---------------------------------------------------------------------------
# Empty document / no chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_empty_chunks_returns_zero():
    store = StubVectorStore()
    tool = ChunkEmbedUpsertTool(
        chunker=StubChunker([]),
        embedder=StubEmbedder([]),
        vector_store=store,
        collection_name="col",
    )
    result = await tool.run({"document": _doc()}, _session())
    assert result == {"doc_id": "doc-1", "chunks_upserted": 0}
    assert store.upserted == []


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_wrong_type_raises():
    tool = ChunkEmbedUpsertTool(
        chunker=StubChunker([]),
        embedder=StubEmbedder([]),
        vector_store=StubVectorStore(),
        collection_name="col",
    )
    with pytest.raises(TypeError, match="Document"):
        await tool.run({"document": "not-a-doc"}, _session())


@pytest.mark.asyncio
async def test_run_missing_key_raises():
    tool = ChunkEmbedUpsertTool(
        chunker=StubChunker([]),
        embedder=StubEmbedder([]),
        vector_store=StubVectorStore(),
        collection_name="col",
    )
    with pytest.raises(KeyError):
        await tool.run({}, _session())


# ---------------------------------------------------------------------------
# Embedder mismatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_embedder_count_mismatch_raises():
    chunk = _chunk("text", 0)
    tool = ChunkEmbedUpsertTool(
        chunker=StubChunker([chunk]),
        embedder=StubEmbedderMismatch(),
        vector_store=StubVectorStore(),
        collection_name="col",
    )
    with pytest.raises(ValueError, match="embeddings"):
        await tool.run({"document": _doc()}, _session())
