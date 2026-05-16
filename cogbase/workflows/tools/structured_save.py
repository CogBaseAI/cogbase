"""structured-save tool — upsert rendered records into a structured collection."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from cogbase.stores import StructuredStoreBase
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import StructuredSaveStepConfig


async def run(
    step: "StructuredSaveStepConfig",
    ctx: dict,
    structured_store: StructuredStoreBase | None,
) -> dict[str, Any]:
    if structured_store is None:
        raise RuntimeError("structured-save requires a structured store")

    records = [render_value(r, ctx) for r in step.records]
    if records:
        await structured_store.save(step.collection, records)
    return {"records": records}
