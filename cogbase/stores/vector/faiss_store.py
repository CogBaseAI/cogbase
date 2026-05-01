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
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np

from cogbase.core.models import Chunk
from cogbase.stores.vector.base import VectorCollectionSchema, VectorStoreBase
from cogbase.stores.filters import Filter, matches

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    logger.exception("faiss_import_failed")
    raise ImportError(
        "faiss-cpu is required for FAISSVectorStore. "
        'Install it with: pip install "cogbase[faiss]"'
    ) from exc


@dataclass
class _CollectionState:
    schema: VectorCollectionSchema
    index: faiss.Index
    chunks: dict[str, Chunk] = field(default_factory=dict)
    id_map: dict[str, int] = field(default_factory=dict)   # chunk_id → faiss int id
    id_rev: dict[int, str] = field(default_factory=dict)   # faiss int id → chunk_id
    next_id: int = 0


class FAISSVectorStore(VectorStoreBase):
    """Vector store backed by per-collection FAISS IndexFlatIP indices.

    Similarity metric: cosine (vectors are L2-normalised before indexing).
    Chunks without an embedding are silently skipped on ``upsert``.

    ``create_collection`` must be called before ``upsert``, ``search``, or
    ``delete`` — operations on an undeclared collection raise ``KeyError``.
    """

    def __init__(self) -> None:
        self._collections: dict[str, _CollectionState] = {}

    # ------------------------------------------------------------------
    # VectorStoreBase interface
    # ------------------------------------------------------------------

    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        """Register a collection. Idempotent — a second call with the same name is a no-op."""
        if schema.name not in self._collections:
            self._collections[schema.name] = _CollectionState(
                schema=schema,
                index=_make_index(schema.dimensions),
            )

    async def delete_collection(self, collection: str) -> None:
        """Remove a collection and all its chunks from memory."""
        self._collections.pop(collection, None)

    async def list_collections(self) -> list[str]:
        return list(self._collections.keys())

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        """Add or replace chunks. Chunks without an embedding are skipped.

        Raises:
            KeyError: If *collection* has not been created via ``create_collection``.
        """
        state = self._get_collection(collection)
        incoming = [c for c in chunks if c.embedding is not None]
        if not incoming:
            return

        dim = len(incoming[0].embedding)  # type: ignore[arg-type]
        if dim != state.schema.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch for collection '{collection}': "
                f"schema declares {state.schema.dimensions}, got {dim}"
            )

        updates = [c for c in incoming if c.chunk_id in state.id_map]
        additions = [c for c in incoming if c.chunk_id not in state.id_map]

        if updates:
            _remove_faiss_ids(state, [state.id_map[c.chunk_id] for c in updates])
            for c in updates:
                del state.id_rev[state.id_map.pop(c.chunk_id)]

        to_add = updates + additions
        vectors = np.array([c.embedding for c in to_add], dtype=np.float32)
        faiss.normalize_L2(vectors)

        faiss_ids = np.arange(state.next_id, state.next_id + len(to_add), dtype=np.int64)
        state.index.add_with_ids(vectors, faiss_ids)

        for chunk, fid in zip(to_add, faiss_ids.tolist()):
            state.chunks[chunk.chunk_id] = chunk
            state.id_map[chunk.chunk_id] = fid
            state.id_rev[fid] = chunk.chunk_id

        state.next_id += len(to_add)

    async def search(
        self,
        collection: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[Chunk]:
        """Return up to ``top_k`` chunks ordered by cosine similarity (highest first).

        Raises:
            KeyError: If *collection* has not been created via ``create_collection``.
        """
        state = self._get_collection(collection)
        if state.index.ntotal == 0:
            return []

        active_filters = filters or []
        k = state.index.ntotal if active_filters else min(top_k, state.index.ntotal)
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)

        _, faiss_ids = state.index.search(q, k)

        results: list[Chunk] = []
        for fid in faiss_ids[0].tolist():
            if fid == -1:
                continue
            chunk_id = state.id_rev.get(fid)
            if chunk_id is None:
                continue
            chunk = state.chunks[chunk_id]
            if active_filters and not matches(chunk.model_dump(), active_filters):
                continue
            results.append(_project_chunk(chunk, fields))
            if len(results) == top_k:
                break
        return results

    async def delete(self, collection: str, doc_id: str) -> None:
        """Remove all chunks belonging to ``doc_id`` and rebuild the index.

        Raises:
            KeyError: If *collection* has not been created via ``create_collection``.
        """
        state = self._get_collection(collection)
        targets = [cid for cid, c in state.chunks.items() if c.doc_id == doc_id]
        if not targets:
            return

        faiss_ids = [state.id_map.pop(cid) for cid in targets]
        for fid in faiss_ids:
            state.id_rev.pop(fid, None)
        for cid in targets:
            state.chunks.pop(cid)

        _rebuild(state)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save(self, path: str | Path) -> None:
        """Persist all collections (FAISS indices + chunk metadata) to *path*.

        *path* is a directory that will be created if it does not exist.  For
        each collection ``<name>`` two files are written:

        * ``<name>.faiss`` — the FAISS index binary
        * ``meta.json``    — schemas, chunk objects, and ID mappings for all collections
        """
        if not self._collections:
            raise RuntimeError("Nothing to save — no collections registered.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync, Path(path))

    def _save_sync(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta: dict = {"collections": {}}
        for name, state in self._collections.items():
            faiss.write_index(state.index, str(path / f"{name}.faiss"))
            meta["collections"][name] = {
                "schema": state.schema.model_dump(),
                "next_id": state.next_id,
                "id_map": state.id_map,
                "id_rev": {str(k): v for k, v in state.id_rev.items()},
                "chunks": {cid: chunk.model_dump() for cid, chunk in state.chunks.items()},
            }
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    async def load(self, path: str | Path) -> None:
        """Load a previously saved store from *path* (a directory written by ``save``)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync, Path(path))

    def _load_sync(self, path: Path) -> None:
        meta: dict = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self._collections = {}
        for name, data in meta["collections"].items():
            schema = VectorCollectionSchema(**data["schema"])
            index = faiss.read_index(str(path / f"{name}.faiss"))
            self._collections[name] = _CollectionState(
                schema=schema,
                index=index,
                chunks={cid: Chunk(**c) for cid, c in data["chunks"].items()},
                id_map=data["id_map"],
                id_rev={int(k): v for k, v in data["id_rev"].items()},
                next_id=data["next_id"],
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def ntotal(self, collection: str) -> int:
        """Number of vectors in *collection*. Returns 0 if the collection does not exist."""
        state = self._collections.get(collection)
        return state.index.ntotal if state is not None else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self, collection: str) -> _CollectionState:
        state = self._collections.get(collection)
        if state is None:
            raise KeyError(
                f"Collection '{collection}' not found. Call create_collection first."
            )
        return state


def _remove_faiss_ids(state: _CollectionState, faiss_ids: list[int]) -> None:
    ids_arr = np.array(faiss_ids, dtype=np.int64)
    state.index.remove_ids(faiss.IDSelectorBatch(ids_arr))


def _rebuild(state: _CollectionState) -> None:
    state.index = _make_index(state.schema.dimensions)
    state.id_map.clear()
    state.id_rev.clear()
    state.next_id = 0

    chunks_with_emb = [c for c in state.chunks.values() if c.embedding is not None]
    if not chunks_with_emb:
        return

    vectors = np.array([c.embedding for c in chunks_with_emb], dtype=np.float32)
    faiss.normalize_L2(vectors)
    faiss_ids = np.arange(len(chunks_with_emb), dtype=np.int64)
    state.index.add_with_ids(vectors, faiss_ids)

    for chunk, fid in zip(chunks_with_emb, faiss_ids.tolist()):
        state.id_map[chunk.chunk_id] = fid
        state.id_rev[fid] = chunk.chunk_id
    state.next_id = len(chunks_with_emb)


def _make_index(dim: int) -> faiss.Index:
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))


def _project_chunk(chunk: Chunk, fields: list[str] | None) -> Chunk:
    if not fields:
        return chunk
    field_set = set(fields)
    return Chunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        embedding=chunk.embedding if "embedding" in field_set else None,
        metadata=chunk.metadata if "metadata" in field_set else {},
    )
