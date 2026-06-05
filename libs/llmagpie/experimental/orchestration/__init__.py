"""Multi-agent orchestration — supervisor / worker pattern.

The supervisor pattern lets one agent decompose a task, delegate to
specialist worker agents via tool-calls, and aggregate their results.
This module provides the high-level :class:`Supervisor` class plus
its building blocks.

See ``MULTI_AGENT_PLAN.md`` at the repo root for the design rationale.
"""

from ._stream import SupervisorChunk
from ._supervisor import Supervisor, SupervisorResult
from ._trace import DelegationTrace
from ._worker import HandoffArgs, WorkerHandle, WorkerResult

__all__ = [
    "DelegationTrace",
    "HandoffArgs",
    "Supervisor",
    "SupervisorChunk",
    "SupervisorResult",
    "WorkerHandle",
    "WorkerResult",
]
