"""llm-structured tool — call an LLM and parse its response against a JSON schema."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from cogbase.core.basemodel_to_schema import cls_json_schema_for_llm
from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema
from cogbase.llms.base import LLMBase
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import WorkflowStepConfig

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


async def run(
    step: "WorkflowStepConfig",
    ctx: dict,
    llm: LLMBase | None,
) -> dict[str, Any]:
    if llm is None:
        raise RuntimeError("llm-structured requires an LLM")
    if not step.prompt:
        raise ValueError("llm-structured step missing 'prompt'")
    if not step.output_schema:
        raise ValueError("llm-structured step missing 'output_schema'")

    system_message = str(render_value(step.prompt, ctx))

    input_values: dict[str, Any] = {
        k: render_value(v, ctx) for k, v in (step.input or {}).items()
    }

    schema_model = build_model_from_json_schema(step.output_schema)
    schema_hint = cls_json_schema_for_llm(schema_model)

    user_message = (
        json.dumps(input_values, default=_json_default, indent=2)
        + f"\n\n---\n\nReturn ONLY valid JSON matching this schema. No markdown fences, no explanation:\n{schema_hint}"
    )

    result = await llm.complete(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    content = result.get("content", "")
    if not content:
        raise ValueError("llm-structured: LLM returned empty response")

    output = schema_model.model_validate_json(content)
    return {"output": output}
