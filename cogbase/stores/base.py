"""Backward-compatible re-exports for store base contracts.

Prefer importing from:
- ``cogbase.stores.structured.base``
- ``cogbase.stores.vector.base``
"""

from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.vector.base import VectorCollectionSchema, VectorStoreBase

__all__ = ["StructuredStoreBase", "VectorCollectionSchema", "VectorStoreBase"]
