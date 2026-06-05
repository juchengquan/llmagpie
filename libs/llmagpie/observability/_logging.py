"""ContextVar-aware logging filter + optional JSON formatter.

Injects fields from the active :class:`RunContext` onto every
:class:`logging.LogRecord` so format strings can reference
``%(run_id)s``, ``%(agent)s``, ``%(worker)s``, ``%(depth)s`` without
KeyError'ing when no run is in flight.

When no context is active, sensible placeholders (``"-"`` for strings,
``0`` for ``depth``) are written, so the same format string works
both inside and outside an llmagpie run.

The optional :class:`JsonFormatter` emits one JSON object per log
line — useful for shipping to log aggregators (Loki, Cloud Logging,
Datadog) that prefer structured fields over regex-parsed text.
"""

from __future__ import annotations

import json
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


# Fields the JSON formatter copies through if present. The
# :class:`RunContextFilter` populates the first four; ``exc_info`` is
# rendered separately via :meth:`Formatter.formatException`.
_JSON_BASE_FIELDS = ("run_id", "agent", "worker", "depth")


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line. Each object has stable keys:

    - ``ts`` — ISO 8601 timestamp (UTC).
    - ``level`` — ``"INFO"`` / ``"WARNING"`` / etc.
    - ``logger`` — the logger name.
    - ``msg`` — the rendered message string.
    - ``run_id`` / ``agent`` / ``worker`` / ``depth`` — from the
      :class:`RunContextFilter`. ``"-"`` / ``0`` when no run is active.
    - ``exc`` — exception traceback string, only when the record had
      ``exc_info``.

    The formatter delegates timestamp conversion to the base class
    (so ``LLMAGPIE_LOG_TZ`` still applies if the base ``Formatter``'s
    converter was overridden). Unknown extras attached to the record
    via ``logger.info(..., extra={...})`` are passed through as keys
    so callers can add structured data without subclassing.
    """

    # Attributes Python's logging library puts on every LogRecord. We
    # don't want these clobbering our top-level keys when we pass
    # caller-supplied ``extra`` through, so we filter them out.
    _STD_ATTRS = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "asctime",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in _JSON_BASE_FIELDS:
            payload[field] = getattr(record, field, "-" if field != "depth" else 0)

        # Caller-supplied ``extra`` — anything on the record that
        # isn't a stdlib attribute or one we already handled.
        for key, value in record.__dict__.items():
            if (
                key in self._STD_ATTRS
                or key in payload
                or key in _JSON_BASE_FIELDS
                or key.startswith("_")
            ):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)
