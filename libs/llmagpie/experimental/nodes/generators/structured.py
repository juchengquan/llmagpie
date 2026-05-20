"""Structured-output helpers — call an LLM, parse its content as JSON,
and validate against a Pydantic schema, with bounded self-repair on
parse failure.

This is provider-agnostic: it sits on top of any :class:`BaseLLMNode`
subclass. Providers that have a native structured-output mode (OpenAI's
``response_format={"type": "json_schema"}``, Ollama's
``format="json"``) can additionally pre-constrain the model — pass
those via ``params`` to :meth:`BaseLLMNode.async_call`."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from ._base import BaseLLMNode

M = TypeVar("M", bound=BaseModel)

# Fenced JSON like ```json {...} ``` — strip the fence before parsing.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


class StructuredOutputError(ValueError):
    """Raised when an LLM response cannot be parsed into the requested
    schema after all repair attempts are exhausted."""

    def __init__(self, message: str, *, last_content: str, last_error: Exception) -> None:
        super().__init__(message)
        self.last_content = last_content
        self.last_error = last_error


def _extract_json_payload(text: str) -> str:
    """Best-effort JSON extraction from free-form LLM output.

    Strips ``` fences if present; otherwise returns the substring from
    the first ``{``/``[`` to the matching last ``}``/``]``. Lets the
    common cases pass even when the model prepends/appends prose.
    """
    if not text:
        return text
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        return fenced.group(1)
    # Heuristic: take the largest brace/bracket region.
    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if not starts:
        return text
    start = min(starts)
    # Walk from the end for matching close.
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    end = text.rfind(close_char)
    if end <= start:
        return text
    return text[start : end + 1]


async def call_with_schema(
    node: BaseLLMNode,
    model: str,
    messages: list[dict[str, Any]],
    schema: type[M],
    *,
    max_repair_attempts: int = 1,
    repair_role: str = "user",
    params: dict[str, Any] | None = None,
) -> M:
    """Call ``node`` and return its response parsed into ``schema``.

    The LLM's terminal ``content`` is parsed as JSON, then fed into
    ``schema.model_validate(...)``. If either step fails, append an
    error-correction message to ``messages`` and retry up to
    ``max_repair_attempts`` times.

    Args:
        node: A :class:`BaseLLMNode` instance (or subclass).
        model: Provider-specific model id.
        messages: OpenAI-style message list. **Mutated in place** on
            repair (an error-correction message gets appended); pass a
            copy if you need to preserve the original.
        schema: Pydantic model class the response should validate against.
        max_repair_attempts: Number of retries after the first failure.
            ``0`` means one shot, no repair.
        repair_role: Role used for the correction message; ``"user"``
            works for all major providers, ``"system"`` works on some.
        params: Forwarded to :meth:`BaseLLMNode.async_call` per-call
            (e.g. ``{"format": "json"}`` for Ollama, OpenAI
            ``response_format``).

    Returns:
        An instance of ``schema`` populated from the LLM's output.

    Raises:
        StructuredOutputError: if all attempts fail to produce a valid
            response. ``last_content`` and ``last_error`` are attached
            for diagnostics.
    """
    last_content = ""
    last_error: Exception | None = None
    attempts = max_repair_attempts + 1

    for attempt in range(attempts):
        # Drive the LLM driver loop; we want only the terminal response.
        last_response = None
        async for response in node.async_call(model=model, messages=messages, params=params):
            last_response = response
        if last_response is None:
            raise StructuredOutputError(
                "node.async_call yielded no responses",
                last_content="",
                last_error=RuntimeError("empty stream"),
            )
        last_content = last_response.content

        try:
            payload = _extract_json_payload(last_content)
            data = json.loads(payload)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            # Append the bad assistant turn so the model sees its own
            # output, then a correction prompt.
            messages.append({"role": "assistant", "content": last_content})
            messages.append(
                {
                    "role": repair_role,
                    "content": (
                        "Your previous response did not parse as the expected schema. "
                        f"Error: {exc}\n\n"
                        f"Return ONLY a JSON object matching this schema (no prose, no fences):\n"
                        f"{json.dumps(schema.model_json_schema())}"
                    ),
                }
            )

    raise StructuredOutputError(
        f"LLM output did not match schema after {attempts} attempt(s): {last_error}",
        last_content=last_content,
        last_error=last_error or RuntimeError("unknown"),
    )
