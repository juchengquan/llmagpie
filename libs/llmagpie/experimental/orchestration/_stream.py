"""Streaming event type for :meth:`Supervisor.stream`.

The supervisor multiplexes its own LLM stream and the active
worker's stream into a single :class:`SupervisorChunk` iterator. The
``source`` + ``event`` + ``worker`` fields let callers route the
chunks to a UI ("Researcher: ...") or aggregate per-agent.

This is the supervisor-explicit-multiplexer approach mentioned in
the plan — sidesteps the LangGraph (#226) and AutoGen (#6136)
hierarchical streaming bugs by never relying on subgraph events
flowing through transparently."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..nodes.generators._base import StreamChunk


class SupervisorChunk(BaseModel):
    """One incremental update from a :meth:`Supervisor.stream` call.

    Attributes:
        source: ``"supervisor"`` for the supervisor's own LLM output,
            ``"worker"`` for tokens forwarded from an active worker.
        worker: Set to the worker name when ``source == "worker"``;
            ``None`` for supervisor chunks.
        event: Optional boundary marker. ``"start"`` precedes a
            worker's stream; ``"end"`` follows it. ``None`` for
            content chunks.
        chunk: The underlying :class:`StreamChunk` from the LLM
            provider. ``None`` for boundary markers.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["supervisor", "worker"]
    worker: str | None = None
    event: Literal["start", "delta", "end"] | None = None
    chunk: StreamChunk | None = None
