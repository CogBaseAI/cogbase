import pytest

from cogbase.stores.structured import InMemoryStructuredStore, SQLiteStructuredStore


@pytest.fixture(params=["memory", "sqlite_file", "sqlite_memory"])
def structured_store(request, tmp_path):
    """Parametrized fixture that yields each StructuredStoreBase implementation."""
    if request.param == "memory":
        return InMemoryStructuredStore()
    elif request.param == "sqlite_file":
        store = SQLiteStructuredStore(tmp_path / "test.db")
        request.addfinalizer(store.close)
        return store
    else:  # sqlite_memory
        store = SQLiteStructuredStore(":memory:")
        request.addfinalizer(store.close)
        return store
