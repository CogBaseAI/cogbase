"""Contract tests for PostgresStructuredStore.

These tests require a running PostgreSQL instance.  Set the
``COGBASE_TEST_PG_DSN`` environment variable before running::

    COGBASE_TEST_PG_DSN=postgresql://localhost/cogbase_test pytest tests/stores/test_postgres_store.py

All contract assertions are imported from ``test_structured_store`` so the
Postgres backend is held to exactly the same standard as SQLite and in-memory.
"""

import pytest

# Re-use all contract tests by importing them into this module's namespace so
# pytest collects them with the postgres_store fixture injected as ``structured_store``.
from tests.stores.test_structured_store import (
    make_fact,
    make_event,
    make_contradiction,
    test_create_collection_is_idempotent,
    test_save_to_unknown_collection_raises,
    test_save_and_query_no_filters,
    test_query_as_deserialises_to_model,
    test_save_upserts_by_id,
    test_eq_filter,
    test_ne_filter,
    test_gte_filter,
    test_lt_filter,
    test_lte_filter,
    test_gt_filter,
    test_in_filter,
    test_not_in_filter,
    test_like_filter_prefix,
    test_like_filter_case_insensitive,
    test_is_null_filter,
    test_is_not_null_filter,
    test_multiple_filters_are_anded,
    test_no_filters_returns_all,
    test_no_match_returns_empty,
    test_json_payload_roundtrip,
    test_contradiction_nested_facts_roundtrip,
    test_boolean_filter,
    test_delete_by_filter,
    test_delete_with_range_filter,
    test_delete_all_with_no_filters,
    test_delete_no_match_is_noop,
    test_custom_collection_with_rich_filters,
    test_add_field_to_existing_collection,
    test_existing_rows_get_null_for_added_field,
    test_remove_field_from_schema_is_ignored_on_read,
    test_migration_is_idempotent,
    test_update_collection_add_field_existing_rows_get_null,
    test_update_collection_add_field_new_rows_can_populate_it,
    test_update_collection_remove_field_data_is_gone,
    test_update_collection_add_and_remove_simultaneously,
    test_update_collection_surviving_fields_data_preserved,
    test_update_collection_unknown_collection_raises,
    test_update_collection_cannot_change_id_field,
    test_update_collection_no_change_is_noop,
)


@pytest.fixture
async def structured_store(postgres_store):
    """Alias so the imported contract tests receive the Postgres store."""
    return postgres_store
