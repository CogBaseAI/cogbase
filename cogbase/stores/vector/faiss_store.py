"""FAISS implementations of VectorStoreBase.

Uses IndexFlatIP (inner product) with L2-normalised vectors, which is equivalent
to cosine similarity — the standard metric for text embeddings.

Install the extra dependency before use:
    pip install "cogbase[faiss]"

Not thread-safe. ``FAISSMemoryVectorStore`` stores data only in memory.
``FAISSVectorStore`` is file-backed and persists each mutation to disk.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np

from cogbase.core.models import Chunk
from cogbase.stores.vector.base import VectorCollectionSchema, VectorStoreBase
from cogbase.stores.vector.chunk_codec import project_chunk
from cogbase.stores.filters import Filter, matches
from cogbase.stores.scope import AppScope

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


class FAISSMemoryVectorStore(VectorStoreBase):
    """In-memory vector store backed by per-collection FAISS IndexFlatIP indices.

    Similarity metric: cosine (vectors are L2-normalised before indexing).
    Chunks without an embedding are silently skipped on ``upsert``.

    ``create_collection`` must be called before ``upsert``, ``search``, or
    ``delete`` — operations on an undeclared collection raise ``KeyError``.
    """

    def __init__(self, scope: AppScope | None = None) -> None:
        super().__init__(scope)
        self._collections: dict[str, _CollectionState] = {}

    # ------------------------------------------------------------------
    # VectorStoreBase interface
    # ------------------------------------------------------------------

    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        """Register a collection. Idempotent — a second call with the same name is a no-op."""
        key = self._c(schema.name)
        if key not in self._collections:
            self._collections[key] = _CollectionState(
                schema=schema,
                index=_make_index(schema.dimensions),
            )
            await self._after_mutation()

    async def delete_collection(self, collection: str) -> None:
        """Remove a collection and all its chunks from memory."""
        if self._collections.pop(self._c(collection), None) is not None:
            await self._after_mutation()

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
        await self._after_mutation()

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
            results.append(project_chunk(chunk, fields))
            if len(results) == top_k:
                break
        return results

    async def delete(self, collection: str, chunk_ids: list[str]) -> None:
        """Remove the chunks identified by ``chunk_ids``.

        ``chunk_id`` values not present in the collection are ignored; an empty
        list is a no-op.

        Raises:
            KeyError: If *collection* has not been created via ``create_collection``.
        """
        if not chunk_ids:
            return
        state = self._get_collection(collection)
        if _remove_chunk_ids(state, chunk_ids):
            await self._after_mutation()

    async def delete_doc(self, collection: str, doc_id: str) -> None:
        """Remove all chunks belonging to ``doc_id``.

        Raises:
            KeyError: If *collection* has not been created via ``create_collection``.
        """
        state = self._get_collection(collection)
        targets = [cid for cid, c in state.chunks.items() if c.doc_id == doc_id]
        if _remove_chunk_ids(state, targets):
            await self._after_mutation()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def ntotal(self, collection: str) -> int:
        """Number of vectors in *collection*. Returns 0 if the collection does not exist."""
        state = self._collections.get(self._c(collection))
        return state.index.ntotal if state is not None else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self, collection: str) -> _CollectionState:
        state = self._collections.get(self._c(collection))
        if state is None:
            raise KeyError(
                f"Collection '{collection}' not found. Call create_collection first."
            )
        return state

    async def _after_mutation(self) -> None:
        """Hook for persistent subclasses."""


class FAISSVectorStore(FAISSMemoryVectorStore):
    """File-backed FAISS vector store that persists every mutation.

    Args:
        path: Directory containing ``meta.json`` and one ``<collection>.faiss``
              file per collection. If omitted, a temporary directory is used.

    Existing data is loaded lazily before the first operation, so callers do not
    need to call ``load`` explicitly.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        dim: int | None = None,
        scope: AppScope | None = None,
    ) -> None:
        super().__init__(scope=scope)
        # Kept for older callers that passed a default FAISS dimension. Per-collection
        # dimensions now come from VectorCollectionSchema.
        _ = dim
        self.path = (
            Path(path)
            if path is not None
            else Path(tempfile.mkdtemp(prefix="cogbase_faiss_"))
        )
        self._loaded = False
        self._persisting = False
        self._dirty = False

    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        await self._ensure_loaded()
        await super().create_collection(schema)

    async def delete_collection(self, collection: str) -> None:
        await self._ensure_loaded()
        await super().delete_collection(collection)

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        await self._ensure_loaded()
        await super().upsert(collection, chunks)

    async def search(
        self,
        collection: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[Chunk]:
        await self._ensure_loaded()
        return await super().search(collection, query, query_embedding, top_k, filters, fields)

    async def delete(self, collection: str, chunk_ids: list[str]) -> None:
        await self._ensure_loaded()
        await super().delete(collection, chunk_ids)

    async def delete_doc(self, collection: str, doc_id: str) -> None:
        await self._ensure_loaded()
        await super().delete_doc(collection, doc_id)

    async def save(self, path: str | Path | None = None) -> None:
        """Persist all collections (FAISS indices + chunk metadata) to disk.

        When *path* is omitted, writes to the store's configured ``path``.  For
        each collection ``<name>`` two files are written:

        * ``<name>.faiss`` — the FAISS index binary
        * ``meta.json``    — schemas, chunk objects, and ID mappings for all collections
        """
        if not self._collections:
            raise RuntimeError("Nothing to save — no collections registered.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync, Path(self.path if path is None else path))

    def _save_sync(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        expected_index_files = {f"{name}.faiss" for name in self._collections}
        for index_file in path.glob("*.faiss"):
            if index_file.name not in expected_index_files:
                index_file.unlink()
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

    async def load(self, path: str | Path | None = None) -> None:
        """Load a previously saved store from disk."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync, Path(self.path if path is None else path))
        self._loaded = True

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

    def ntotal(self, collection: str) -> int:
        if not self._loaded and (self.path / "meta.json").exists():
            self._load_sync(self.path)
            self._loaded = True
        return super().ntotal(collection)  # parent already calls self._c()

    async def _after_mutation(self) -> None:
        # Mark the store dirty. If a save is already in flight, the running loop
        # below will pick up this change on its next iteration — mutations that
        # land while _save_sync is persisting collection-by-collection are not
        # dropped, which would otherwise lose data on shutdown.
        self._dirty = True
        if self._persisting:
            return
        self._persisting = True
        try:
            loop = asyncio.get_running_loop()
            while self._dirty:
                self._dirty = False
                await loop.run_in_executor(None, self._save_sync, self.path)
        finally:
            self._persisting = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if (self.path / "meta.json").exists():
            await self.load()
        self._loaded = True


def _remove_faiss_ids(state: _CollectionState, faiss_ids: list[int]) -> None:
    ids_arr = np.array(faiss_ids, dtype=np.int64)
    state.index.remove_ids(faiss.IDSelectorBatch(ids_arr))


def _remove_chunk_ids(state: _CollectionState, chunk_ids: list[str]) -> list[int]:
    """Drop the given ``chunk_ids`` from *state* and its FAISS index.

    Unknown ``chunk_id`` values are ignored. Returns the FAISS ids that were
    actually removed (empty when nothing matched), so callers can skip the
    persistence hook on a no-op.
    """
    faiss_ids: list[int] = []
    for cid in chunk_ids:
        fid = state.id_map.pop(cid, None)
        if fid is None:
            continue
        state.id_rev.pop(fid, None)
        state.chunks.pop(cid, None)
        faiss_ids.append(fid)
    if faiss_ids:
        _remove_faiss_ids(state, faiss_ids)
    return faiss_ids


def _make_index(dim: int) -> faiss.Index:
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))


