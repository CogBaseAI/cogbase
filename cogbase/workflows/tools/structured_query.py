"""structured-query tool — query a structured collection with EQ filters."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from cogbase.stores import StructuredStoreBase
from cogbase.stores.filters import Col
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import WorkflowStepConfig


async def run(
    step: "WorkflowStepConfig",
    ctx: dict,
    structured_store: StructuredStoreBase | None,
) -> dict[str, Any]:
    if structured_store is None:
        raise RuntimeError("structured-query requires a structured store")
    if not step.collection:
        raise ValueError("structured-query step missing 'collection'")

    filters = [
        Col(field) == render_value(val_template, ctx)
        for field, val_template in (step.filters or {}).items()
    ]
    records = await structured_store.query(step.collection, filters or None)
    return {"records": records}
