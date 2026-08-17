"""structured-query tool — query a structured collection with EQ filters."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from cogbase.stores import StructuredStoreBase
from cogbase.stores.filters import Col
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import StructuredQueryStepConfig

logger = logging.getLogger(__name__)


async def run(
    step: "StructuredQueryStepConfig",
    ctx: dict,
    structured_store: StructuredStoreBase | None,
) -> dict[str, Any]:
    if structured_store is None:
        raise RuntimeError("structured-query requires a structured store")

    values = {
        field: render_value(val_template, ctx)
        for field, val_template in step.filters.items()
    }
    filters = [Col(field) == value for field, value in values.items()]
    records = await structured_store.query(step.collection, filters or None)
    logger.info(
        "workflow.tool.structured_query collection=%s filters=%d records=%d",
        step.collection, len(filters), len(records),
    )
    # Checked after the log line so the count that failed is on record. The
    # rendered filter values go in the message rather than the templates: the
    # usual cause is a value that rendered to something no row carries, and the
    # template alone does not show what it became.
    if len(records) < step.min_records:
        raise RuntimeError(
            f"structured-query step {step.id!r}: collection {step.collection!r} "
            f"returned {len(records)} records with filters {values!r}, below the "
            f"step's min_records={step.min_records}. The step declares an empty "
            "result to be a precondition failure rather than an empty workload, "
            "so the run fails here instead of completing with nothing done."
        )
    return {"records": records}
