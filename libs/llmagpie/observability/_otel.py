"""GenAI-semconv span helpers.

Four context managers — :func:`agent_span`, :func:`handoff_span`,
:func:`tool_span`, :func:`chat_span` — plus a :func:`set_llm_attributes`
helper that stamps :class:`LLMResponse` data onto an in-flight span.

All four are *no-ops when OpenTelemetry isn't installed*. When OTel
is installed they emit spans via the global tracer, which is set up
by :mod:`llmagpie.core.opentelemetry` when ``OTEL_COLLECTOR_ENDPOINT``
is configured — and is OTel's default NoOpTracer otherwise, so tests
can swap in :class:`InMemorySpanExporter` without touching env vars.

Attribute conventions follow the OpenTelemetry GenAI semantic
conventions (``gen_ai.*``) plus the OpenInference
``openinference.span.kind`` enum that Phoenix / Arize / LangSmith /
Langfuse / MLflow render. The fully-qualified attribute names live in
constants below so call sites stay terse and a future semconv shift
is a single-file edit.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

try:
    from opentelemetry import trace as _trace
    from opentelemetry.trace.status import Status as _Status
    from opentelemetry.trace.status import StatusCode as _StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the no-otel install
    _OTEL_AVAILABLE = False
    _trace = None  # type: ignore[assignment]
    _Status = None  # type: ignore[assignment,misc]
    _StatusCode = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from collections.abc import Iterator


# --- OpenInference span-kind values ----------------------------------------

SPAN_KIND_AGENT = "AGENT"
SPAN_KIND_TOOL = "TOOL"
SPAN_KIND_LLM = "LLM"
SPAN_KIND_CHAIN = "CHAIN"


# --- Attribute keys --------------------------------------------------------

ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_SESSION_ID = "session.id"

# GenAI semconv (OTel) — the values that Phoenix/Arize/LangSmith read.
ATTR_GEN_AI_SYSTEM = "gen_ai.system"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
ATTR_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
ATTR_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# llmagpie-specific attrs for handoffs — there's no semconv value yet.
ATTR_HANDOFF_SOURCE = "llmagpie.handoff.source"
ATTR_HANDOFF_TARGET = "llmagpie.handoff.target"
ATTR_HANDOFF_DEPTH = "llmagpie.handoff.depth"
ATTR_HANDOFF_TASK_PREVIEW = "llmagpie.handoff.task_preview"


# --- Internal: tracer + null-span fallback ---------------------------------


def _tracer() -> Any:
    """Return the configured tracer, or ``None`` if OTel isn't
    importable. When OTel is importable but no provider was
    explicitly set, this returns OTel's default NoOpTracer — spans
    are created but discarded, which is the correct fallback."""
    if not _OTEL_AVAILABLE:
        return None
    return _trace.get_tracer("llmagpie")


class _NullSpan:
    """Stand-in for an OTel span when OTel isn't installed. Accepts
    every set / record / status call as a no-op so call sites don't
    need to special-case the missing-OTel branch."""

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_attributes(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_status(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


_NULL_SPAN = _NullSpan()


def _session_id() -> str | None:
    """Pull ``run_id`` off the active :class:`RunContext` so every
    GenAI span auto-correlates to the run, without each call site
    threading it through. Lazy-imported to avoid an import cycle:
    ``observability.__init__`` imports this module."""
    from ._context import current

    ctx = current()
    return ctx.run_id if ctx is not None else None


# --- Public span helpers ----------------------------------------------------


@contextmanager
def agent_span(
    *,
    agent_name: str,
    is_supervisor: bool = False,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Open an ``invoke_agent`` span for the duration of a
    :meth:`Agent.run` / :meth:`Supervisor.run` call.

    Stamps:
        - ``openinference.span.kind = "AGENT"``
        - ``gen_ai.agent.name = agent_name``
        - ``gen_ai.operation.name = "invoke_agent"``
        - ``session.id = <run_id>`` from the active :class:`RunContext`
        - any caller-supplied attrs

    No-op when OpenTelemetry isn't installed.
    """
    tracer = _tracer()
    if tracer is None:
        yield _NULL_SPAN
        return

    span_name = f"{'supervise' if is_supervisor else 'invoke_agent'} {agent_name}"
    with tracer.start_as_current_span(span_name) as span:
        _set_attrs(
            span,
            {
                ATTR_SPAN_KIND: SPAN_KIND_AGENT,
                ATTR_GEN_AI_AGENT_NAME: agent_name,
                ATTR_GEN_AI_OPERATION_NAME: "invoke_agent",
            },
        )
        _set_session_id(span)
        if attributes:
            _set_attrs(span, attributes)
        try:
            yield span
        except Exception as exc:
            _record_exc(span, exc)
            raise


@contextmanager
def handoff_span(*, source: str, target: str, task: str, depth: int) -> Iterator[Any]:
    """Open a ``handoff`` span around a supervisor → worker dispatch.

    Stamps:
        - ``openinference.span.kind = "CHAIN"`` (no GenAI semconv for
          handoffs yet; CHAIN renders cleanly across all backends)
        - ``gen_ai.operation.name = "handoff"``
        - ``llmagpie.handoff.{source,target,depth,task_preview}``
        - ``session.id``

    Becomes the parent of the worker's ``agent_span`` so the trace
    tree shows the delegation chain.
    """
    tracer = _tracer()
    if tracer is None:
        yield _NULL_SPAN
        return

    with tracer.start_as_current_span(f"handoff {source}→{target}") as span:
        _set_attrs(
            span,
            {
                ATTR_SPAN_KIND: SPAN_KIND_CHAIN,
                ATTR_GEN_AI_OPERATION_NAME: "handoff",
                ATTR_HANDOFF_SOURCE: source,
                ATTR_HANDOFF_TARGET: target,
                ATTR_HANDOFF_DEPTH: depth,
                ATTR_HANDOFF_TASK_PREVIEW: task[:120],
            },
        )
        _set_session_id(span)
        try:
            yield span
        except Exception as exc:
            _record_exc(span, exc)
            raise


@contextmanager
def tool_span(*, tool_name: str) -> Iterator[Any]:
    """Open an ``execute_tool`` span around a tool invocation.

    Stamps ``openinference.span.kind = "TOOL"`` and
    ``gen_ai.tool.name``. The caller is responsible for calling
    :meth:`Span.record_exception` if the tool raises (via the yielded
    span); we don't intercept the tool's exception here because the
    framework wraps tool errors into return values upstream.
    """
    tracer = _tracer()
    if tracer is None:
        yield _NULL_SPAN
        return

    with tracer.start_as_current_span(f"execute_tool {tool_name}") as span:
        _set_attrs(
            span,
            {
                ATTR_SPAN_KIND: SPAN_KIND_TOOL,
                ATTR_GEN_AI_TOOL_NAME: tool_name,
                ATTR_GEN_AI_OPERATION_NAME: "execute_tool",
            },
        )
        _set_session_id(span)
        try:
            yield span
        except Exception as exc:
            _record_exc(span, exc)
            raise


@contextmanager
def chat_span(*, model: str | None = None, system: str | None = None) -> Iterator[Any]:
    """Open a ``chat`` span around a single LLM round-trip.

    The caller is expected to call :func:`set_llm_attributes` (passing
    the yielded span) after the LLM call returns, so token counts and
    finish reasons get stamped from the :class:`LLMResponse`.
    """
    tracer = _tracer()
    if tracer is None:
        yield _NULL_SPAN
        return

    with tracer.start_as_current_span(f"chat {model or '?'}") as span:
        _set_attrs(
            span,
            {
                ATTR_SPAN_KIND: SPAN_KIND_LLM,
                ATTR_GEN_AI_OPERATION_NAME: "chat",
            },
        )
        if system:
            span.set_attribute(ATTR_GEN_AI_SYSTEM, system)
        if model:
            span.set_attribute(ATTR_GEN_AI_REQUEST_MODEL, model)
        _set_session_id(span)
        try:
            yield span
        except Exception as exc:
            _record_exc(span, exc)
            raise


def set_llm_attributes(
    span: Any,
    *,
    model: str | None = None,
    usage: Any = None,
    finish_reason: str | None = None,
    system: str | None = None,
) -> None:
    """Stamp post-call GenAI attributes onto an in-flight chat span.

    Safe to call on the :class:`_NullSpan` returned by the no-op
    branch — every set is a no-op there.
    """
    if span is None:
        return
    if system:
        span.set_attribute(ATTR_GEN_AI_SYSTEM, system)
    if model:
        span.set_attribute(ATTR_GEN_AI_RESPONSE_MODEL, model)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", 0)
        if prompt:
            span.set_attribute(ATTR_GEN_AI_USAGE_INPUT_TOKENS, prompt)
        if completion:
            span.set_attribute(ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, completion)
    if finish_reason:
        span.set_attribute(ATTR_GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])


# --- Internal helpers ------------------------------------------------------


def _set_attrs(span: Any, attrs: dict[str, Any]) -> None:
    """``set_attributes`` rejects ``None`` values on some backends —
    filter them out, then bulk-set. Falls back to per-key
    ``set_attribute`` so OTel versions without ``set_attributes`` work."""
    clean = {k: v for k, v in attrs.items() if v is not None}
    if not clean:
        return
    bulk = getattr(span, "set_attributes", None)
    if callable(bulk):
        bulk(clean)
    else:
        for k, v in clean.items():
            span.set_attribute(k, v)


def _set_session_id(span: Any) -> None:
    sid = _session_id()
    if sid:
        span.set_attribute(ATTR_SESSION_ID, sid)


def _record_exc(span: Any, exc: BaseException) -> None:
    rec = getattr(span, "record_exception", None)
    if callable(rec):
        rec(exc)
    if _Status is not None and _StatusCode is not None:
        span.set_status(_Status(_StatusCode.ERROR, str(exc)))
