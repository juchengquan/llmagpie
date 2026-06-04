"""Delegation trace — a tree of `WorkerHandle.dispatch()` calls.

Captured by :class:`Supervisor` during a run so callers can inspect
who was called, with what task, how long it took, and what each
sub-call cost in tokens. Trees nest naturally when a worker is
itself a :class:`Supervisor`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..nodes.generators._base import LLMUsage


class DelegationTrace(BaseModel):
    """One node in the delegation tree.

    Attributes:
        worker: Name of the worker that was invoked. ``"<root>"`` on
            the supervisor's top-level node.
        task: The task string the supervisor handed off (or the
            original user message at the root).
        depth: Nesting depth — 0 at the supervisor, 1 at its workers,
            2 at workers-of-workers, etc.
        started_at: Wall-clock seconds (``time.monotonic()``).
        ended_at: Wall-clock seconds at completion, or ``None`` if
            still in flight when serialized (rare; usually the trace
            is read after the run finishes).
        usage: Token usage attributed to this call (NOT including
            children — sum the tree if you want cumulative).
        error: Stringified error if the worker raised. ``None`` on
            success.
        children: Nested delegations spawned during this call.
    """

    model_config = ConfigDict(extra="forbid")

    worker: str
    task: str
    depth: int
    started_at: float
    ended_at: float | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error: str | None = None
    children: list[DelegationTrace] = Field(default_factory=list)

    def format(self, indent: int = 0) -> str:
        """Pretty-print the trace as an indented tree, useful for debugging.

        Each line shows ``[duration] worker: task (tokens)`` with
        children indented two spaces further.
        """
        pad = "  " * indent
        dur = (self.ended_at - self.started_at) if self.ended_at else None
        dur_str = f"{dur:.2f}s" if dur is not None else "..."
        head = f"{pad}[{dur_str}] {self.worker}"
        if self.task:
            preview = self.task if len(self.task) <= 60 else self.task[:57] + "..."
            head += f": {preview}"
        if self.usage.total_tokens:
            head += f"  ({self.usage.total_tokens} tok)"
        if self.error:
            head += f"  ERROR: {self.error}"
        lines = [head]
        for child in self.children:
            lines.append(child.format(indent=indent + 1))
        return "\n".join(lines)

    def cumulative_usage(self) -> LLMUsage:
        """Sum token usage across this node and all descendants."""
        total = LLMUsage(
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            total_tokens=self.usage.total_tokens,
        )
        for child in self.children:
            child_total = child.cumulative_usage()
            total.prompt_tokens += child_total.prompt_tokens
            total.completion_tokens += child_total.completion_tokens
            total.total_tokens += child_total.total_tokens
        return total
