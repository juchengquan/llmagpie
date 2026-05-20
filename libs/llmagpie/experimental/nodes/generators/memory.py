"""Conversation memory — persist chat history across invocations.

The framework's per-session ``input_state`` is cleared at the end of
every ``invoke()``. For multi-turn conversations the caller needs
history that outlives a single call; that's what :class:`MemoryNode`
provides.

A :class:`MemoryStore` is the pluggable backend (in-memory by default;
write your own for SQLite/Postgres/Redis). :class:`MemoryNode` wraps an
inner :class:`BaseLLMNode` and, per ``thread_id``:

1. Loads the saved history.
2. Prepends it to the caller-supplied ``messages``.
3. Calls the inner LLM through the standard tool-calling loop.
4. Appends the new user/assistant exchange back to the store.

Trimming policy (``max_messages``) keeps the conversation bounded so
older context naturally falls out the back."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import ConfigDict, Field

from ._base import BaseLLMNode, LLMResponse


@runtime_checkable
class MemoryStore(Protocol):
    """Minimal async per-thread message store. Implementations should
    be safe to call concurrently from multiple coroutines."""

    async def get(self, thread_id: str) -> list[dict[str, Any]]: ...
    async def append(self, thread_id: str, messages: list[dict[str, Any]]) -> None: ...
    async def clear(self, thread_id: str) -> None: ...


class InMemoryStore:
    """Process-local thread → message-list store. Good for tests and
    short-lived processes; lost on restart."""

    def __init__(self) -> None:
        self._threads: dict[str, list[dict[str, Any]]] = {}

    async def get(self, thread_id: str) -> list[dict[str, Any]]:
        return list(self._threads.get(thread_id, []))

    async def append(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        self._threads.setdefault(thread_id, []).extend(messages)

    async def clear(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)


def _trim(messages: list[dict[str, Any]], max_messages: int | None) -> list[dict[str, Any]]:
    """Keep the last ``max_messages`` while always preserving every
    leading ``system`` message (those are usually instructions, not
    conversation turns)."""
    if max_messages is None or len(messages) <= max_messages:
        return messages
    system_prefix = []
    body = messages
    while body and body[0].get("role") == "system":
        system_prefix.append(body[0])
        body = body[1:]
    keep = max(0, max_messages - len(system_prefix))
    return system_prefix + body[-keep:] if keep else system_prefix


class MemoryNode(BaseLLMNode):
    """Memory-augmented LLM node.

    Wraps any :class:`BaseLLMNode` (``inner``) and a :class:`MemoryStore`
    (``store``). On each ``async_call``, conversations are scoped to
    ``thread_id`` (passed in ``params``). Defaults to ``"default"`` so
    single-thread use just works.

    Args:
        inner: The wrapped LLM node.
        store: Pluggable persistence (:class:`InMemoryStore` by default).
        max_messages: If set, trim history to the last N messages
            (system messages always preserved) before sending to the
            inner LLM. ``None`` means no trim.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    inner: Any = Field(default=None, description="The wrapped BaseLLMNode")
    store: Any = Field(default=None, description="The MemoryStore instance")
    max_messages: int | None = None

    def bind_tools(self, tools: list[Any]) -> MemoryNode:
        self.inner.bind_tools(tools)
        self.tools_node = self.inner.tools_node
        return self

    def _format_tools_for_provider(self) -> list[dict] | None:
        return self.inner._format_tools_for_provider()

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        # Pop the thread_id from kwargs so it doesn't leak into the
        # provider call; default to "default" for single-thread use.
        thread_id: str = kwargs.pop("thread_id", "default")

        history = await self.store.get(thread_id)
        combined = _trim(history + messages, self.max_messages)
        response = await self.inner._complete(model, combined, **kwargs)

        # Persist the new turns. We store the user/system messages that
        # came in on THIS call plus the assistant's response, but NOT
        # the historical messages we just loaded (those are already in
        # the store).
        new_turns: list[dict[str, Any]] = list(messages)
        assistant_turn: dict[str, Any] = {
            "role": response.role,
            "content": response.content,
        }
        if response.tool_calls:
            assistant_turn["tool_calls"] = response.tool_calls
        new_turns.append(assistant_turn)
        await self.store.append(thread_id, new_turns)

        return response
