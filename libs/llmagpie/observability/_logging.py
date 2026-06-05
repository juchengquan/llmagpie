"""ContextVar-aware logging filter.

Injects fields from the active :class:`RunContext` onto every
:class:`logging.LogRecord` so format strings can reference
``%(run_id)s``, ``%(agent)s``, ``%(worker)s``, ``%(depth)s`` without
KeyError'ing when no run is in flight.

When no context is active, sensible placeholders (``"-"`` for strings,
``0`` for ``depth``) are written, so the same format string works
both inside and outside an llmagpie run.
"""

from __future__ import annotations

import logging

from ._context import current


class RunContextFilter(logging.Filter):
    """Stamp the active :class:`RunContext` fields onto every record.

    Safe to attach to any handler; it never blocks records (always
    returns True) and runs in O(1) — a single ContextVar.get plus a
    handful of attribute assignments.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = current()
        if ctx is not None:
            # Truncate run_id for terminal readability; full id is on
            # the context for callers that want it.
            record.run_id = ctx.run_id[:8] if ctx.run_id else "-"
            record.agent = ctx.agent or "-"
            record.worker = ctx.worker or "-"
            record.depth = ctx.depth
        else:
            record.run_id = "-"
            record.agent = "-"
            record.worker = "-"
            record.depth = 0
        return True
