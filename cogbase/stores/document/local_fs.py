"""Local filesystem document store."""

from __future__ import annotations

import asyncio
import pathlib
import shutil

from cogbase.stores.document.base import DocumentStoreBase
from cogbase.stores.scope import AppScope


class LocalFSDocumentStore(DocumentStoreBase):
    """Stores document text as UTF-8 files under a root directory.

    Each document is written to ``<root>/<collection>/<doc_id>`` (intermediate
    directories are created automatically).  Hierarchical doc_ids (e.g.
    ``"2024/q1/doc-1"``) produce a matching directory tree.

    Args:
        root:  Directory that will hold all document files.  Created on first
               ``save`` if it does not exist.
        scope: Optional scope that prefixes collection names, preventing collisions
               when multiple applications share the same root directory.
    """

    def __init__(self, root: str | pathlib.Path, scope: AppScope | None = None) -> None:
        super().__init__(scope)
        self._root = pathlib.Path(root).resolve()

    def _path(self, collection: str, doc_id: str) -> pathlib.Path:
        candidate = (self._root / self._c(collection) / doc_id).resolve()
        if not str(candidate).startswith(str(self._root)):
            raise ValueError(f"collection/doc_id {collection!r}/{doc_id!r} escapes the store root")
        return candidate

    async def save(self, collection: str, doc_id: str, content: str) -> None:
        path = self._path(collection, doc_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write, path, content)

    async def load(self, collection: str, doc_id: str) -> str:
        path = self._path(collection, doc_id)
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, path.read_text, "utf-8")
        except FileNotFoundError:
            raise KeyError(doc_id)

    async def delete(self, collection: str, doc_id: str) -> None:
        path = self._path(collection, doc_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._unlink, path)

    async def delete_collection(self, collection: str) -> None:
        col_dir = (self._root / self._c(collection)).resolve()
        if not str(col_dir).startswith(str(self._root)):
            raise ValueError(f"collection {collection!r} escapes the store root")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._rmtree, col_dir)

    async def exists(self, collection: str, doc_id: str) -> bool:
        path = self._path(collection, doc_id)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, path.is_file)

    async def save_bytes(self, collection: str, doc_id: str, content: bytes) -> None:
        path = self._path(collection, doc_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_bytes, path, content)

    async def load_bytes(self, collection: str, doc_id: str) -> bytes:
        path = self._path(collection, doc_id)
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, path.read_bytes)
        except FileNotFoundError:
            raise KeyError(doc_id)

    # -- sync helpers -------------------------------------------------------

    @staticmethod
    def _write(path: pathlib.Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_bytes(path: pathlib.Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    @staticmethod
    def _unlink(path: pathlib.Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _rmtree(path: pathlib.Path) -> None:
        if path.exists():
            shutil.rmtree(path)
