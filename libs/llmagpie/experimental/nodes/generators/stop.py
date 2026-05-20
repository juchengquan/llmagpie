"""Stop-condition factories for the :class:`BaseLLMNode` tool-call loop.

Each helper returns a ``Callable[[LLMResponse], bool]`` that can be
plugged into ``BaseLLMNode.stop_condition`` (or
``Agent(stop_condition=...)``). When the callable returns True after a
provider round-trip, the loop exits without dispatching further tool
calls or making another LLM call."""

from __future__ import annotations

import re
from collections.abc import Callable

from ._base import LLMResponse

StopCondition = Callable[[LLMResponse], bool]


def stop_on_content_match(pattern: str | re.Pattern, *, flags: int = 0) -> StopCondition:
    """Stop when the response's ``content`` matches ``pattern``.

    Useful for the canonical "agent says DONE" pattern::

        Agent(..., stop_condition=stop_on_content_match(r"\\bDONE\\b"))

    Args:
        pattern: A regex (string or compiled) searched against
            ``response.content`` with :func:`re.search` semantics.
        flags: Optional ``re`` flags applied when compiling a string
            pattern; ignored for an already-compiled pattern.
    """
    if isinstance(pattern, str):
        compiled = re.compile(pattern, flags)
    else:
        compiled = pattern

    def _stop(response: LLMResponse) -> bool:
        return bool(compiled.search(response.content or ""))

    return _stop


def stop_on_tool_name(name: str) -> StopCondition:
    """Stop the moment the LLM asks to call a specific tool.

    Use for agents that designate one tool as the "final answer"
    sentinel (e.g. ``finish``, ``submit_answer``)::

        Agent(..., stop_condition=stop_on_tool_name("submit_answer"))
    """

    def _stop(response: LLMResponse) -> bool:
        return any((tc.get("function") or {}).get("name") == name for tc in response.tool_calls)

    return _stop


def stop_on_finish_reason(*reasons: str) -> StopCondition:
    """Stop when the provider's ``finish_reason`` is one of ``reasons``.

    Example: ``stop_on_finish_reason("length", "content_filter")``
    aborts on a truncated response instead of letting the loop spin.
    """
    reason_set = set(reasons)

    def _stop(response: LLMResponse) -> bool:
        return response.finish_reason in reason_set

    return _stop


def any_of(*conditions: StopCondition) -> StopCondition:
    """Compose multiple stop conditions with OR semantics."""

    def _stop(response: LLMResponse) -> bool:
        return any(c(response) for c in conditions)

    return _stop
