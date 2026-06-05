"""Debug-mode runtime capture — JSONL tape of every LLM round-trip.

A ContextVar-backed :class:`TapeWriter` lives in
:data:`_active_tape`. When set (by :class:`Agent` / :class:`Supervisor`
with ``debug=True``), :meth:`BaseLLMNode._complete_traced` appends each
(request, response) pair to it. When unset (the default), no overhead:
``current_tape()`` returns ``None`` and the LLM path doesn't touch
the filesystem.

The tape format mirrors :class:`RecordReplayLLMNode`'s: newline-
delimited JSON, one entry per recorded call::

    {"request": {"model": ..., "messages": [...], "kwargs": {...}},
     "response": {<LLMResponse.model_dump()>},
     "timestamp": "2026-06-05T12:34:56.789Z",
     "agent": "planner"}

Per-run isolation falls out of the ContextVar design: when a supervisor
opens a capture and dispatches to a worker that also has ``debug=True``,
the worker's Agent.run pushes its own tape on the stack, the worker's
LLM calls write only to the worker's tape, and the supervisor's tape
resumes once the worker exits. If the worker has ``debug=False``, the
supervisor's tape stays active and the worker's calls land there.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from llmagpie.experimental.nodes.generators._base import LLMResponse


class TapeWriter:
    """Appends one JSON entry per call to a JSONL file. Creates the
    parent directory on first write; safe to share across an entire
    run (writes are append-only and small)."""

    __slots__ = ("_count", "agent_label", "path")

    def __init__(self, path: Path, *, agent_label: str | None = None) -> None:
        self.path = path
        self.agent_label = agent_label
        self._count = 0

    @property
    def count(self) -> int:
        """Number of entries written so far. Useful for tests."""
        return self._count

    def write(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        response: LLMResponse,
    ) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "agent": self.agent_label,
            "request": {
                "model": model,
                "messages": messages,
                "kwargs": kwargs,
            },
            "response": response.model_dump(),
        }
        # Lazy parent-dir create — keeps construction cheap, supports
        # ``debug_dir`` paths that don't exist yet.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        self._count += 1


_active_tape: ContextVar[TapeWriter | None] = ContextVar("llmagpie_active_tape", default=None)


def current_tape() -> TapeWriter | None:
    """Return the :class:`TapeWriter` for the active capture, or
    ``None`` if no agent currently has ``debug=True`` in flight."""
    return _active_tape.get()


@contextmanager
def capture_to(path: Path | str, *, agent_label: str | None = None) -> Iterator[TapeWriter]:
    """Open a capture context: every subsequent LLM round-trip inside
    the ``with`` block appends to ``path``.

    Nestable — an inner ``capture_to`` replaces the outer for the
    duration of its block, then the outer resumes. The framework
    relies on this for supervisor/worker tape isolation when both
    have ``debug=True``.
    """
    writer = TapeWriter(Path(path), agent_label=agent_label)
    token = _active_tape.set(writer)
    try:
        yield writer
    finally:
        _active_tape.reset(token)


def resolve_debug_path(*, debug_dir: Path | str | None, run_id: str, agent_label: str) -> Path:
    """Build the per-run tape file path::

        <debug_dir>/<run_id_8>__<agent_label>.jsonl

    Truncates the run_id to 8 chars for filesystem readability;
    correlation back to logs / spans is still unambiguous.
    Defaults to ``./.llmagpie-debug/`` when ``debug_dir`` is unset.
    """
    base = Path(debug_dir) if debug_dir is not None else Path("./.llmagpie-debug")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in agent_label)
    return base / f"{run_id[:8]}__{safe}.jsonl"
