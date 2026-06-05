"""Exception enrichment — attach the live :class:`RunContext` to any
exception that bubbles out of an llmagpie entry point.

The framework doesn't introduce a new base exception class. Instead,
the existing ones (:class:`BudgetExceededError`,
:class:`StructuredOutputError`, plus arbitrary
caller-tool / provider-SDK exceptions) gain a single ``run_context``
attribute when they propagate through a wrapped entry point.

This is idempotent: an inner frame that already attached its own
context wins. The outer frame's :func:`attach_context` call is a
no-op when ``exc.run_context`` is already set. That way the post-mortem
sees the *innermost* state the framework knew about, not the outermost.
"""

from __future__ import annotations

from ._context import RunContext, current


def attach_context(exc: BaseException, ctx: RunContext | None = None) -> BaseException:
    """Stash the current :class:`RunContext` onto ``exc`` as
    ``exc.run_context``.

    Idempotent — if ``exc`` already carries a context (set by a deeper
    frame), the existing one is preserved. ``ctx`` overrides the
    auto-detected current context when supplied (useful for the
    Supervisor case where we want to capture the in-flight delegation
    trace explicitly).
    """
    if getattr(exc, "run_context", None) is None:
        captured = ctx if ctx is not None else current()
        if captured is not None:
            try:
                exc.run_context = captured  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                # Some exceptions (e.g. C-extension subclasses) reject
                # attribute setting. The post-mortem helper degrades
                # gracefully on missing attributes, so just swallow.
                pass
    return exc
