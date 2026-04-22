"""FAISS implementation of VectorStoreBase.

Uses IndexFlatIP (inner product) with L2-normalised vectors, which is equivalent
to cosine similarity — the standard metric for text embeddings.

Install the extra dependency before use:
    pip install "cogbase[faiss]"

Not thread-safe. Embeddings are stored in-memory alongside the FAISS index;
data is lost when the process exits unless you call ``save`` / ``load``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import logging

import numpy as np

from cogbase.core.models import Chunk
from cogbase.stores.base import VectorCollectionSchema, VectorStoreBase

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    logger.exception("faiss_import_failed")
    raise ImportError(
        "faiss-cpu is required for FAISSVectorStore. "
        'Install it with: pip install "cogbase[faiss]"'
    ) from exc


class FAISSVectorStore(VectorStoreBase):
    """Vector store backed by a FAISS IndexFlatIP index.

    Similarity metric: cosine (vectors are L2-normalised before indexing).
    Chunks without an embedding are silently skipped on ``upsert``.

    Args:
        dim: Embedding dimension. If ``None``, inferred from the first ``upsert``.
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim: int | None = dim
        self._index: faiss.Index | None = None
        if dim is not None:
            self._index = _make_index(dim)

        # chunk_id → Chunk (source of truth for all stored data)
        self._chunks: dict[str, Chunk] = {}
        # chunk_id → FAISS integer id
        self._id_map: dict[str, int] = {}
        # FAISS integer id → chunk_id
        self._id_rev: dict[int, str] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # VectorStoreBase interface
    # ------------------------------------------------------------------

    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        """No-op — FAISS is schema-free; the index is created lazily on first upsert."""

    async def delete_collection(self, collection: str) -> None:
        """Reset the store, discarding all chunks and the FAISS index."""
        self._dim = None
        self._index = None
        self._chunks.clear()
        self._id_map.clear()
        self._id_rev.clear()
        self._next_id = 0

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        """Add or replace chunks. Chunks without an embedding are skipped."""
        incoming = [c for c in chunks if c.embedding is not None]
        if not incoming:
            return

        dim = len(incoming[0].embedding)  # type: ignore[arg-type]
        self._ensure_index(dim)

        # Separate updates (chunk_id already indexed) from new additions
        updates = [c for c in incoming if c.chunk_id in self._id_map]
        additions = [c for c in incoming if c.chunk_id not in self._id_map]

        if updates:
            # Remove stale FAISS entries for updated chunks, then re-add below
            self._remove_faiss_ids([self._id_map[c.chunk_id] for c in updates])
            for c in updates:
                del self._id_rev[self._id_map.pop(c.chunk_id)]

        to_add = updates + additions
        vectors = np.array([c.embedding for c in to_add], dtype=np.float32)
        faiss.normalize_L2(vectors)

        faiss_ids = np.arange(self._next_id, self._next_id + len(to_add), dtype=np.int64)
        self._index.add_with_ids(vectors, faiss_ids)  # type: ignore[union-attr]

        for chunk, fid in zip(to_add, faiss_ids.tolist()):
            self._chunks[chunk.chunk_id] = chunk
            self._id_map[chunk.chunk_id] = fid
            self._id_rev[fid] = chunk.chunk_id

        self._next_id += len(to_add)

    async def search(self, collection: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """Return up to ``top_k`` chunks ordered by cosine similarity (highest first)."""
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        _, faiss_ids = self._index.search(query, k)

        results: list[Chunk] = []
        for fid in faiss_ids[0].tolist():
            if fid == -1:  # FAISS sentinel for no result
                continue
            chunk_id = self._id_rev.get(fid)
            if chunk_id is not None:
                results.append(self._chunks[chunk_id])
        return results

    async def delete(self, collection: str, doc_id: str) -> None:
        """Remove all chunks belonging to ``doc_id`` and rebuild the index."""
        targets = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        if not targets:
            return

        faiss_ids = [self._id_map.pop(cid) for cid in targets]
        for fid in faiss_ids:
            self._id_rev.pop(fid, None)
        for cid in targets:
            self._chunks.pop(cid)

        self._rebuild()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save(self, path: str | Path) -> None:
        """Persist the full store (FAISS index + chunk metadata) to *path*.

        *path* is a directory that will be created if it does not exist.  Two
        files are written inside it:

        * ``index.faiss`` — the FAISS index binary
        * ``meta.json``   — chunk objects, ID mappings, and index metadata
        """
        if self._index is None:
            raise RuntimeError("Nothing to save — index is empty.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync, Path(path))

    def _save_sync(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        meta: dict = {
            "dim": self._dim,
            "next_id": self._next_id,
            "id_map": self._id_map,
            # JSON requires string keys; int keys are restored on load
            "id_rev": {str(k): v for k, v in self._id_rev.items()},
            "chunks": {cid: chunk.model_dump() for cid, chunk in self._chunks.items()},
        }
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    async def load(self, path: str | Path) -> None:
        """Load a previously saved store from *path* (a directory written by ``save``)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync, Path(path))

    def _load_sync(self, path: Path) -> None:
        self._index = faiss.read_index(str(path / "index.faiss"))
        meta: dict = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self._dim = meta["dim"]
        self._next_id = meta["next_id"]
        self._id_map = meta["id_map"]
        self._id_rev = {int(k): v for k, v in meta["id_rev"].items()}
        self._chunks = {cid: Chunk(**data) for cid, data in meta["chunks"].items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal if self._index is not None else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = _make_index(dim)
        elif dim != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: index expects {self._dim}, got {dim}"
            )

    def _remove_faiss_ids(self, faiss_ids: list[int]) -> None:
        ids_arr = np.array(faiss_ids, dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids_arr))  # type: ignore[union-attr]

    def _rebuild(self) -> None:
        """Rebuild the FAISS index from the current _chunks dict."""
        if not self._index:
            return

        self._index = _make_index(self._dim)  # type: ignore[arg-type]
        self._id_map.clear()
        self._id_rev.clear()
        self._next_id = 0

        chunks_with_emb = [c for c in self._chunks.values() if c.embedding is not None]
        if not chunks_with_emb:
            return

        vectors = np.array([c.embedding for c in chunks_with_emb], dtype=np.float32)
        faiss.normalize_L2(vectors)
        faiss_ids = np.arange(len(chunks_with_emb), dtype=np.int64)
        self._index.add_with_ids(vectors, faiss_ids)

        for chunk, fid in zip(chunks_with_emb, faiss_ids.tolist()):
            self._id_map[chunk.chunk_id] = fid
            self._id_rev[fid] = chunk.chunk_id
        self._next_id = len(chunks_with_emb)


def _make_index(dim: int) -> faiss.Index:
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))
