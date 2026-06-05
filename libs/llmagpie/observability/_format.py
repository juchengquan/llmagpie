"""Human-readable formatting of llmagpie exceptions and traces.

:func:`format_error` produces a multi-line post-mortem string given
any exception that flowed through a wrapped entry point. It composes
the exception's own message with the :class:`RunContext` fields and,
when present, the :class:`DelegationTrace` tree.

Callers use it alongside (not in place of) the regular traceback::

    try:
        result = await supervisor.run("...")
    except BudgetExceededError as exc:
        print(format_error(exc))
        raise
"""

from __future__ import annotations

from typing import Any


def format_error(exc: BaseException, *, include_trace: bool = True) -> str:
    """Return a multi-line, human-readable post-mortem of ``exc``.

    Output shape::

        BudgetExceededError: supervisor exceeded max_tokens_per_run (15234 > 10000)
          run_id:     7af3c1b8
          agent:      writer
          supervisor: planner
          worker:     writer (depth 2)
          thread:     thread_alpha

        Delegation trace:
          [1.20s] planner: Write a brief on ...  (15234 tok)
            [0.40s] researcher: Find sources  (5100 tok)
            [0.78s] writer: Draft summary  (8200 tok)

    If the exception was raised outside any framework entry point (no
    ``run_context`` attached) the function still returns the original
    message — just without the context block.

    Args:
        exc: Any exception. If it carries a :class:`RunContext`
            (attached by :func:`attach_context`), the context block is
            rendered. Otherwise only the header line is returned.
        include_trace: When True (the default), include the
            :class:`DelegationTrace` tree if one is on the context.
    """
    lines = [f"{type(exc).__name__}: {exc}"]

    ctx = getattr(exc, "run_context", None)
    if ctx is None:
        return lines[0]

    info = _context_lines(ctx)
    if info:
        lines.append("")
        lines.extend(info)

    if include_trace and getattr(ctx, "delegation_trace", None) is not None:
        trace = ctx.delegation_trace
        format_method = getattr(trace, "format", None)
        if callable(format_method):
            lines.append("")
            lines.append("Delegation trace:")
            lines.append(format_method())

    # Surface a couple of well-known auxiliary attributes that the
    # framework's own exceptions carry, when present, so users see
    # them inline without extra digging.
    extras = _extra_attrs(exc)
    if extras:
        lines.append("")
        lines.extend(extras)

    return "\n".join(lines)


def format_trace(ctx_or_trace: Any) -> str:
    """Render a :class:`DelegationTrace` (or a :class:`RunContext`
    carrying one) as the indented tree string.

    Convenience for callers that want the trace without an exception —
    e.g. logging the supervisor's trace after a successful run.
    Returns an empty string if no trace is reachable.
    """
    trace = getattr(ctx_or_trace, "delegation_trace", ctx_or_trace)
    format_method = getattr(trace, "format", None)
    if callable(format_method):
        return format_method()
    return ""


def _context_lines(ctx: Any) -> list[str]:
    """Render the ``run_id`` / ``agent`` / ``supervisor`` / ``worker`` /
    ``thread`` block. Skips fields that are unset so the post-mortem
    stays terse on simple Agent runs (no supervisor, no worker)."""
    rows: list[tuple[str, str]] = []

    run_id = getattr(ctx, "run_id", None)
    if run_id:
        rows.append(("run_id", str(run_id)[:8]))

    agent = getattr(ctx, "agent", None)
    if agent:
        rows.append(("agent", str(agent)))

    supervisor = getattr(ctx, "supervisor", None)
    if supervisor and supervisor != agent:
        rows.append(("supervisor", str(supervisor)))

    worker = getattr(ctx, "worker", None)
    depth = getattr(ctx, "depth", 0)
    if worker:
        rows.append(("worker", f"{worker} (depth {depth})"))
    elif depth:
        rows.append(("depth", str(depth)))

    thread_id = getattr(ctx, "thread_id", None)
    if thread_id:
        rows.append(("thread", str(thread_id)))

    if not rows:
        return []

    width = max(len(k) for k, _ in rows)
    return [f"  {k:<{width}} : {v}" for k, v in rows]


def _extra_attrs(exc: BaseException) -> list[str]:
    """Surface budget / structured-output fields if present. Kept small
    on purpose — only attributes the framework itself sets."""
    out: list[str] = []
    e: Any = exc
    if hasattr(e, "budget_dimension") and hasattr(e, "budget_limit"):
        usage = getattr(e, "usage_so_far", None)
        used_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        if used_tokens is not None:
            out.append(f"  budget: {used_tokens} / {e.budget_limit} ({e.budget_dimension})")
    last_content = getattr(e, "last_content", None)
    if last_content:
        snippet = str(last_content)
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        out.append(f"  last_content: {snippet!r}")
    return out
