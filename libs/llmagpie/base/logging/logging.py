import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import cast
from zoneinfo import ZoneInfo

from llmagpie.observability._logging import RunContextFilter

# Timezone for log timestamps. Honors ``LLMAGPIE_LOG_TZ`` (any IANA
# zone name); defaults to UTC. Override at process start:
#
#     export LLMAGPIE_LOG_TZ=Asia/Singapore
#
# Falls back to UTC if the supplied zone can't be resolved (e.g. an
# alpine container without tzdata).
try:
    _LOG_TZ = ZoneInfo(os.environ.get("LLMAGPIE_LOG_TZ", "UTC"))
except Exception:
    _LOG_TZ = ZoneInfo("UTC")


class CustomFormatter(logging.Formatter):
    """Formatter that renders timestamps in :data:`_LOG_TZ`."""

    def converter(self, timestamp: float | None) -> datetime.datetime:  # type: ignore[override]
        """Convert epoch seconds to a tz-aware datetime in the
        configured timezone.

        Args:
            timestamp: Epoch seconds; falls back to "now" when ``None``
                (matches stdlib :func:`time.localtime` behavior).
        """
        dt = datetime.datetime.fromtimestamp(timestamp or 0.0, tz=datetime.UTC)
        return dt.astimezone(_LOG_TZ)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None):
        """Format the record's timestamp; ISO 8601 when no ``datefmt``."""
        dt = self.converter(cast(int, record.created))
        if datefmt:
            formatted_timestamp = dt.strftime(datefmt)
        else:
            try:
                formatted_timestamp = dt.isoformat(timespec="milliseconds")
            except TypeError:
                formatted_timestamp = dt.isoformat()
        return formatted_timestamp


def DefaultLogger(name: str, handlers: list[logging.Handler] | None = None, *args, **kwargs):
    """Initialize (or fetch) a logger by ``name``. Idempotent — calling
    twice returns the same logger without duplicating handlers.

    A :class:`RunContextFilter` is attached at the logger level (not
    per-handler) so every record gets the correlation fields before it
    reaches any handler — including handlers attached later by tooling
    like pytest's ``caplog`` fixture.
    """
    flag = name in logging.root.manager.loggerDict
    logger = logging.getLogger(name, *args, **kwargs)
    logger.setLevel(logging.INFO)

    if not any(isinstance(f, RunContextFilter) for f in logger.filters):
        logger.addFilter(RunContextFilter())

    if not logger.hasHandlers():
        for handler in handlers or []:
            logger.addHandler(handler)
    if not flag:
        logger.debug(f"Logger initialized: {name}")

    return logger


# Format includes RunContext correlation fields (``run_id``, ``agent``,
# ``worker``) so users can grep a multi-agent log by run id without
# parsing structure. ``RunContextFilter`` populates the fields with
# ``-`` placeholders when no run is in flight, so the format string is
# always satisfied.
LOGGING_FORMAT = (
    '[%(asctime)s] run_id=%(run_id)s agent=%(agent)s worker=%(worker)s '
    'level=%(levelname)s name="%(name)s" filename="%(filename)s" '
    'lineno=%(lineno)d msg="%(message)s"'
)
Formatter = CustomFormatter(LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S,%f")


def _make_handler(handler: logging.Handler) -> logging.Handler:
    """Attach the standard level + formatter to ``handler``. The
    :class:`RunContextFilter` lives on the logger (see
    :func:`DefaultLogger`) so it covers handlers attached later as
    well."""
    handler.setLevel(logging.INFO)
    handler.setFormatter(Formatter)
    return handler


def get_or_create_logger(logger_name: str = "default", file_path: str | None = None):
    """Get or create the named logger, wired with the stream handler
    (and optionally a rotating file handler when ``LOG_DIR`` is set)."""
    streamhandler = _make_handler(logging.StreamHandler())

    if not file_path:
        log_path = os.getenv("LOG_DIR")
        if log_path:
            os.makedirs(log_path, exist_ok=True)
            file_path = f"{log_path}/info.log"

            filehandler = _make_handler(
                TimedRotatingFileHandler(filename=file_path, interval=1, when="d")
            )

            logger = DefaultLogger(name=logger_name, handlers=[streamhandler, filehandler])
        else:
            logger = DefaultLogger(name=logger_name, handlers=[streamhandler])
        return logger
    return DefaultLogger(name=logger_name, handlers=[streamhandler])
