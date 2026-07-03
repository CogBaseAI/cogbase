"""llm-structured tool — call an LLM and parse its response against a JSON schema."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

import jsonschema

from cogbase.llms.base import LLMBase
from cogbase.llms.summarization import estimate_messages_tokens
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import LLMStructuredStepConfig

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


_MAX_RETRIES = 2

# Fraction of the model's context window the rendered input may occupy before the
# step refuses the call. The remainder is headroom for the structured output the
# model must still generate, plus slack for the rough (~chars/4) token estimate.
#
# This step feeds its whole input to the model in one call: its input is often a
# holistic judgment over an upstream query result (e.g. "find contradictions among
# these facts"), which cannot be chunked without either losing cross-item
# relationships or losing fidelity. So on overflow we fail fast with a located,
# actionable error rather than truncating, mis-batching, or letting the provider
# throw a 400 several retries deep — the author is the only party who can safely
# reduce the input (narrow the upstream filters, or distill each item in a foreach
# map step before this judgment).
_INPUT_BUDGET_RATIO = 0.8


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

    window = llm.context_window()
    budget = int(window * _INPUT_BUDGET_RATIO)
    input_tokens = estimate_messages_tokens(messages)
    if input_tokens > budget:
        logger.error(
            "llm_structured.input_over_budget step=%s input_tokens=%d budget=%d window=%d",
            step.id,
            input_tokens,
            budget,
            window,
        )
        raise ValueError(
            f"llm-structured step {step.id!r}: rendered input is ~{input_tokens} tokens, "
            f"over the ~{budget}-token budget ({int(_INPUT_BUDGET_RATIO * 100)}% of the "
            f"{window}-token context window, leaving room for the model's output). This "
            f"step passes its whole input to the model in a single call and cannot chunk a "
            f"holistic task safely. Reduce the input: narrow the upstream query's filters, "
            f"or restructure the workflow (e.g. distill each item with a foreach map step "
            f"before this judgment)."
        )

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
            logger.info(
                "workflow.tool.llm_structured.ok attempt=%d/%d output_keys=%s",
                attempt + 1,
                _MAX_RETRIES + 1,
                list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
            )
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
