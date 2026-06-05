"""Worker handle — exposes an :class:`Agent` as a tool the supervisor can call.

A :class:`WorkerHandle` is a :class:`BaseNode` subclass so its tool
schema (name + description + arg schema) surfaces to the supervisor's
LLM via the standard OpenAI tool-call protocol. The supervisor's
driver loop intercepts calls to worker handles BEFORE they hit the
regular :class:`ToolsNode` dispatch path — workers are called async
(via :meth:`WorkerHandle.dispatch`), regular tools go through the
standard sync ThreadPoolExecutor dispatch.

The two paths are separated because workers are themselves agents
(they run their own LLM calls) and need async dispatch, parent state
(messages, depth, scratchpad), and structured usage roll-up.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llmagpie.base.node import BaseNode, MakeNode
from llmagpie.observability import current_context, derive, handoff_span, push

from ..nodes.generators._base import LLMUsage

# Tool-call name prefix the supervisor uses to identify a worker call.
# E.g. a worker registered with name="researcher" surfaces to the LLM
# as a tool named "transfer_to_researcher".
HANDOFF_PREFIX = "transfer_to_"


class HandoffArgs(BaseModel):
    """Schema for the supervisor's tool-call arguments when delegating.

    Validated at the dispatch boundary; rejection is surfaced to the
    LLM as a tool-error message rather than raised, so the supervisor
    can retry without crashing.
    """

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        description=(
            "A self-contained task description. The worker has no memory of prior "
            "conversation; everything it needs must be in this field."
        )
    )
    context_hint: str | None = Field(
        default=None,
        description="Optional short context the supervisor wants the worker to keep in mind.",
    )


class WorkerResult(BaseModel):
    """Structured result returned by a worker invocation. Wrapped into
    a tool-result message the supervisor's LLM sees on the next turn."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    worker: str
    content: str = ""
    parsed: Any = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error: str | None = None


@MakeNode.from_class(
    func_name="async_call",
    outputs={"content": str, "error": str, "usage": dict},
)
class WorkerHandle(BaseNode):
    """An :class:`Agent` exposed as a tool the supervisor's LLM can call.

    Constructed via :meth:`Agent.as_worker`. The handle carries the
    inner agent plus the context-handoff configuration; the actual
    dispatch happens inside the supervisor's driver loop, which calls
    :meth:`invoke` with the parent state.

    The ``async_call`` method is a fallback for direct invocation
    (outside a supervisor) and runs the worker in ``task_only`` mode
    with no parent context.

    Args (constructor):
        agent: The wrapped :class:`Agent`.
        context_handoff: What the worker sees on input. ``"task_only"``
            (default) is the most defensive — worker sees only the
            task string plus its own system prompt. ``"task_plus_history"``
            includes the last ``history_window`` supervisor messages
            with role-flipping. ``"shared_scratchpad"`` passes a
            structured scratchpad the supervisor maintains.
        history_window: Used when ``context_handoff="task_plus_history"``.
        persistent_thread: When False (default), each invocation uses
            a fresh ``thread_id`` for the worker's memory store — no
            cross-delegation accumulation. Set True for "team
            historian" workers that should remember across calls.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    agent: Any = None  # :class:`Agent` — typed Any because Agent isn't pydantic.
    context_handoff: Literal["task_only", "task_plus_history", "shared_scratchpad"] = "task_only"
    history_window: int = 6
    persistent_thread: bool = False

    def _generate_description_openai(self) -> dict:
        """Surface this handle to the LLM with the ``transfer_to_`` prefix
        so the model recognizes delegation as distinct from regular tools.
        Description is augmented with a literal enumeration of the worker's
        own tools (when present), so the supervisor knows what the worker
        is capable of."""
        tool_name = f"{HANDOFF_PREFIX}{self.name}"
        desc = self.description or f"Delegate a task to the `{self.name}` agent."
        worker_tools = _list_worker_tools(self.agent)
        if worker_tools:
            desc = f"{desc} Available tools: {', '.join(worker_tools)}."
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": desc,
                "parameters": HandoffArgs.model_json_schema(),
            },
        }

    async def async_call(self, task: str, context_hint: str | None = None) -> dict:
        """Fallback entry point for direct invocation outside a supervisor.

        Returns a dict matching the node's declared outputs schema.
        In a supervisor context this is bypassed — the supervisor's
        driver loop calls :meth:`dispatch` directly.
        """
        result = await self.dispatch(
            task=task,
            context_hint=context_hint,
            parent_messages=[],
            depth=1,
            scratchpad=None,
        )
        return {
            "content": result.content,
            "error": result.error or "",
            "usage": result.usage.model_dump(),
        }

    async def dispatch(
        self,
        *,
        task: str,
        context_hint: str | None = None,
        parent_messages: list[dict[str, Any]],
        depth: int,
        scratchpad: dict[str, Any] | None,
    ) -> WorkerResult:
        """Dispatch the worker. Builds messages per ``context_handoff``,
        drives the inner agent, returns a structured :class:`WorkerResult`.

        Never raises — errors are captured into :attr:`WorkerResult.error`
        so the supervisor can recover (the LLM is the recovery loop)."""
        messages = self._build_messages(task, context_hint, parent_messages, scratchpad)

        thread_id = (
            "default" if self.persistent_thread else f"_worker_{self.name}_{depth}_{id(messages)}"
        )

        # Worker frame inherits run_id / supervisor / delegation_trace
        # from the supervisor and stamps its own ``worker`` name and
        # ``depth`` so log lines / traces emitted inside the worker's
        # Agent.run() are attributable.
        ctx = derive(worker=self.name, depth=depth)
        parent = current_context()
        source = (parent.supervisor or parent.agent or "?") if parent is not None else "?"
        with (
            push(ctx),
            handoff_span(source=source, target=self.name, task=task, depth=depth),
        ):
            try:
                # Drive the worker's loop. We use _drive (not run) because we already
                # built the messages list and we want the raw LLMResponse + usage back
                # without going through the schema-repair branch (the worker's own
                # response_schema, if set, still applies inside agent.run()).
                agent_result = await self.agent.run(messages, thread_id=thread_id)
            except Exception as e:
                return WorkerResult(
                    worker=self.name,
                    content="",
                    error=f"{type(e).__name__}: {e}",
                )

        return WorkerResult(
            worker=self.name,
            content=agent_result.content,
            parsed=agent_result.parsed,
            usage=agent_result.usage,
        )

    def _build_messages(
        self,
        task: str,
        context_hint: str | None,
        parent_messages: list[dict[str, Any]],
        scratchpad: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the worker's input message list per the handoff mode."""
        if self.context_handoff == "task_only":
            user_content = task if context_hint is None else f"{task}\n\nContext: {context_hint}"
            return [{"role": "user", "content": user_content}]

        if self.context_handoff == "task_plus_history":
            tail = _summarize_tail(parent_messages, n=self.history_window)
            user_content = task if context_hint is None else f"{task}\n\nContext: {context_hint}"
            return [*tail, {"role": "user", "content": user_content}]

        if self.context_handoff == "shared_scratchpad":
            sp = scratchpad or {}
            blob = json.dumps(sp, default=str)
            user_content = (
                f"<task>{task}</task>\n<scratchpad>{blob}</scratchpad>"
                if context_hint is None
                else (
                    f"<task>{task}</task>\n<context>{context_hint}</context>\n"
                    f"<scratchpad>{blob}</scratchpad>"
                )
            )
            return [{"role": "user", "content": user_content}]

        raise ValueError(f"Unknown context_handoff: {self.context_handoff!r}")


def _summarize_tail(messages: list[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    """Take the last ``n`` non-system messages and role-flip the
    supervisor's assistant turns into annotated user-side context.

    The worker sees ``<supervisor_message>...</supervisor_message>``-wrapped
    content rather than raw assistant turns, so it treats them as background
    rather than as turns to continue from.
    """
    non_system = [m for m in messages if m.get("role") != "system"]
    tail = non_system[-n:] if n > 0 else []
    flipped: list[dict[str, Any]] = []
    for m in tail:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant":
            flipped.append(
                {
                    "role": "user",
                    "content": f"<supervisor_message>{content}</supervisor_message>",
                }
            )
        elif role == "tool":
            # Tool results from the supervisor's history — wrap so the worker
            # knows it's prior tool output, not a fresh result for it.
            flipped.append(
                {
                    "role": "user",
                    "content": f"<prior_tool_result>{content}</prior_tool_result>",
                }
            )
        else:
            flipped.append(dict(m))
    return flipped


def _list_worker_tools(agent: Any) -> list[str]:
    """Walk the agent's wrapped LLM chain looking for a bound ToolsNode,
    return the tool names. Empty list if no tools / no agent."""
    if agent is None:
        return []
    node = getattr(agent, "_llm", None)
    while node is not None:
        tools_node = getattr(node, "tools_node", None)
        if tools_node is not None:
            return [t.name for t in tools_node.tools]
        node = getattr(node, "inner", None)
    return []
