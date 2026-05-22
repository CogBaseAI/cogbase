"""llm-structured tool — call an LLM and parse its response against a JSON schema."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

import jsonschema

from cogbase.llms.base import LLMBase
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import LLMStructuredStepConfig

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


_MAX_RETRIES = 2


async def run(
    step: "LLMStructuredStepConfig",
    ctx: dict,
    llm: LLMBase | None,
) -> dict[str, Any]:
    if llm is None:
        raise RuntimeError("llm-structured requires an LLM")

    schema: dict = json.loads(step.output_schema)
    system_message = (
        str(render_value(step.prompt, ctx))
        + "\n\nReturn ONLY a JSON object, not markdown and not the schema."
        + "\nThe JSON object must validate against this JSON Schema:\n"
        + step.output_schema
    )

    input_values: dict[str, Any] = {
        k: render_value(v, ctx) for k, v in step.input.items()
    }

    user_message = json.dumps(input_values, default=_json_default, indent=2)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            await asyncio.sleep(0.2 * 2 ** (attempt - 1))

        result = await llm.complete(messages, temperature=0.0)
        content = result.get("content", "")
        if not content:
            raise ValueError("llm-structured: LLM returned empty response")

        try:
            parsed = json.loads(content)
            jsonschema.validate(instance=parsed, schema=schema)
            return {"output": parsed}
        except Exception as exc:
            last_exc = exc
            logger.error(
                "llm_structured.parse_failed attempt=%d/%d error=%s, content=%s",
                attempt + 1,
                _MAX_RETRIES + 1,
                exc,
                content,
            )

    raise ValueError("llm-structured: failed to parse LLM response after retries") from last_exc
