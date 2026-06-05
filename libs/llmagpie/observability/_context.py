"""Per-run correlation state for logs, traces, exceptions, and debug tapes.

A single :class:`RunContext` lives in a :mod:`contextvars` ContextVar so
it propagates across ``await`` and :func:`asyncio.create_task` without
any explicit threading. Cross-thread propagation (e.g. through
:class:`concurrent.futures.ThreadPoolExecutor`) requires explicit
:func:`contextvars.copy_context` — see :class:`ToolsNode` for the
pattern.

The four observability surfaces all read from the same ``RunContext``:

- exception enrichment (:mod:`._errors`) reads :func:`current` to stamp
  context onto raised exceptions;
- the OTel helpers (:mod:`._otel`, Phase 2) use ``run_id`` as the
  ``gen_ai.session.id`` span attribute;
- the log filter (:mod:`._logging`) injects ``run_id`` / ``agent`` /
  ``worker`` / ``depth`` into every log record;
- the debug-mode tape (:mod:`._capture`, Phase 3) is named
  ``<run_id>.jsonl``.

That single point of truth lets users trace a single user request from
log → trace → exception → tape via one correlation id.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterator


class RunContext(BaseModel):
    """Correlation state attached to one logical run.

    Created at the outermost entry point (an :class:`Agent.run` /
    :class:`Supervisor.run` call) and inherited by every nested call
    via :func:`derive`. Inner frames override the fields they own
    (``agent``, ``worker``, ``depth``) while leaving ``run_id`` and
    parent-supplied fields intact.

    Attributes:
        run_id: Stable identifier for the entire run; shared across all
            frames so logs / traces / exceptions / tapes correlate.
        agent: Name of the innermost agent currently executing.
        supervisor: Name of the outermost supervisor, if a supervisor
            is in flight; ``None`` for a bare :class:`Agent.run` call.
        worker: Name of the current worker, when inside
            :meth:`WorkerHandle.dispatch`.
        depth: Supervisor delegation depth. 0 at the top-level
            supervisor, 1 at its workers, etc.
        thread_id: Memory thread id, when the agent has a memory store
            attached.
        delegation_trace: The supervisor's :class:`DelegationTrace`
            root, when a supervisor is in flight. Typed ``Any`` here to
            avoid a circular import — ``orchestration/_trace.py``
            imports from ``base`` and we don't want a return-trip.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    agent: str | None = None
    supervisor: str | None = None
    worker: str | None = None
    depth: int = 0
    thread_id: str | None = None
    delegation_trace: Any = None


_run_ctx: ContextVar[RunContext | None] = ContextVar("llmagpie_run_ctx", default=None)


def current() -> RunContext | None:
    """Return the active :class:`RunContext`, or ``None`` if no run is
    in flight. Safe to call at module-import time and from any
    thread."""
    return _run_ctx.get()


def derive(**overrides: Any) -> RunContext:
    """Build a child :class:`RunContext` inheriting from the current
    one, with field overrides.

    Used at every framework entry point: :meth:`Agent.run`,
    :meth:`Supervisor.run`, :meth:`WorkerHandle.dispatch`. The first
    frame (no parent) gets a fresh ``run_id``; nested frames inherit
    the existing ``run_id`` and override only the fields they own.
    """
    parent = current()
    if parent is None:
        return RunContext(**overrides)
    base = parent.model_dump()
    base.update(overrides)
    return RunContext(**base)


@contextmanager
def push(ctx: RunContext) -> Iterator[RunContext]:
    """Activate ``ctx`` as the current :class:`RunContext` for the
    duration of the ``with`` block. The previous context is restored
    on exit even if the body raises."""
    token = _run_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _run_ctx.reset(token)
