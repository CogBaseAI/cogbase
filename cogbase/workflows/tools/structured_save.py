"""structured-save tool — upsert rendered records into a structured collection."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from cogbase.stores import StructuredStoreBase
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import StructuredSaveStepConfig

logger = logging.getLogger(__name__)


async def run(
    step: "StructuredSaveStepConfig",
    ctx: dict,
    structured_store: StructuredStoreBase | None,
) -> dict[str, Any]:
    if structured_store is None:
        raise RuntimeError("structured-save requires a structured store")

    if step.records_from:
        records = render_value(step.records_from, ctx)
        if not isinstance(records, list):
            raise RuntimeError(
                "structured-save records_from must resolve to a list, got "
                f"{type(records).__name__}"
            )
    else:
        records = [render_value(r, ctx) for r in step.records]
    if records:
        await structured_store.save(step.collection, records)
    logger.info(
        "workflow.tool.structured_save collection=%s records=%d",
        step.collection, len(records),
    )
    return {"records": records}
