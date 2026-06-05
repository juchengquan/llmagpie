"""Observability primitives for llmagpie.

Public surface:

- :class:`RunContext` — per-run correlation state, ContextVar-backed.
- :func:`current_context` — read the active :class:`RunContext` (or
  ``None`` outside of a run).
- :func:`format_error` — human-readable post-mortem of a framework
  exception, including the delegation trace when present.
- :func:`format_trace` — render a delegation trace as an indented
  tree string.

The :func:`attach_context` helper and the :class:`RunContextFilter`
log filter are used by the framework's own entry points; most user
code shouldn't need to call them directly.
"""

from ._capture import TapeWriter, capture_to, current_tape, resolve_debug_path
from ._context import RunContext, current, derive, push
from ._errors import attach_context
from ._format import format_error, format_trace
from ._logging import JsonFormatter, RunContextFilter
from ._otel import (
    agent_span,
    chat_span,
    handoff_span,
    set_llm_attributes,
    tool_span,
)

# `current_context` is the public alias; `current` stays as the short
# internal name used by framework code.
current_context = current

__all__ = [
    "JsonFormatter",
    "RunContext",
    "RunContextFilter",
    "TapeWriter",
    "agent_span",
    "attach_context",
    "capture_to",
    "chat_span",
    "current_context",
    "current_tape",
    "derive",
    "format_error",
    "format_trace",
    "handoff_span",
    "push",
    "resolve_debug_path",
    "set_llm_attributes",
    "tool_span",
]
