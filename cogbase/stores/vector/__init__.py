from cogbase.stores.vector.base import VectorCollectionSchema, VectorStoreBase
from cogbase.stores.vector.faiss_store import FAISSMemoryVectorStore, FAISSVectorStore

def __getattr__(name: str):
    if name == "PGVectorStore":
        from cogbase.stores.vector.pgvector_store import PGVectorStore
        return PGVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "VectorCollectionSchema",
    "VectorStoreBase",
    "FAISSVectorStore",
    "FAISSMemoryVectorStore",
    "PGVectorStore",
]
