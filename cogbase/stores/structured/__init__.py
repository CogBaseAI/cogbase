from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.structured.sqlite import SQLiteStructuredStore

def __getattr__(name: str):
    if name == "PostgresStructuredStore":
        from cogbase.stores.structured.postgres import PostgresStructuredStore
        return PostgresStructuredStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["StructuredStoreBase", "InMemoryStructuredStore", "PostgresStructuredStore", "SQLiteStructuredStore"]
