"""High-level agent abstraction.

Composes :class:`BaseLLMNode` with the existing memory, cache, tool,
and structured-output building blocks into a single class so the
common case stops requiring users to wire the parts themselves.

Composition order (outermost wraps innermost):

    MemoryNode -> CachedLLMNode -> raw provider node

So the cache key includes the loaded history (the memory layer
prepends history before the cache lookup), and the cache key is
stable across identical conversations across processes.

If the caller passes ``tools``, they are bound at the outer (memory)
layer so the framework dispatches the tool-call loop with history in
context — the inner cache treats each model→provider round-trip as a
distinct entry."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmagpie.base.node import BaseNode

from .nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from .nodes.generators.cache import CacheBackend, CachedLLMNode
from .nodes.generators.memory import MemoryNode, MemoryStore
from .nodes.generators.structured import StructuredOutputError, _extract_json_payload

M = TypeVar("M", bound=BaseModel)


class AgentResult(BaseModel):
    """The terminal result of an :meth:`Agent.run` call.

    Attributes:
        content: The assistant's final text content (after any tool
            iterations settle).
        tool_calls: Any tool calls in the final LLM response (usually
            empty if the loop converged; non-empty only if
            ``max_tool_iterations`` was hit).
        parsed: If the agent was constructed with ``response_schema``,
            the validated Pydantic instance; otherwise ``None``.
        usage: Total token usage across all provider round-trips in
            this run (summed across tool-call iterations).
        raw: The final :class:`LLMResponse` for callers who want the
            full provider payload.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    content: str
    tool_calls: list[dict] = Field(default_factory=list)
    parsed: BaseModel | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    raw: LLMResponse


class Agent:
    """High-level wrapper around :class:`BaseLLMNode`.

    Composes optional memory, cache, tools, and structured-output
    validation into a single ``run(user_message, ...)`` entry point.
    Most LLM apps want exactly this: pick a provider, give it a system
    prompt, optionally give it tools and memory, and call ``run``.

    Args:
        llm: A :class:`BaseLLMNode` subclass (any provider). Memory and
            cache, if requested, wrap around this.
        model: Provider-specific model identifier (forwarded to every
            ``_complete`` call).
        system_prompt: Optional system instruction prepended to the
            messages on every call.
        tools: Optional list of tool nodes for the LLM to call. Bound
            via :meth:`BaseLLMNode.bind_tools` on the outermost wrapper.
        memory_store: Optional :class:`MemoryStore`. When provided,
            conversation history persists per ``thread_id`` across
            calls. ``None`` (default) means stateless single-turn use.
        cache: Optional :class:`CacheBackend`. When provided, identical
            ``(model, messages, params)`` round-trips are served from
            the cache.
        cache_ttl: Per-entry TTL in seconds for the cache; ``None``
            means no expiry.
        response_schema: Optional Pydantic model class. When set,
            ``run`` returns the parsed instance under
            :attr:`AgentResult.parsed` and self-repairs up to
            ``repair_attempts`` times on parse failure.
        max_tool_iterations: Bound on the tool-calling loop (forwarded
            to the inner LLM's ``max_tool_iterations``).
        repair_attempts: Number of self-repair retries when
            ``response_schema`` validation fails.
        name: Optional name for the agent (used in logging).
    """

    def __init__(
        self,
        llm: BaseLLMNode,
        model: str,
        *,
        system_prompt: str | None = None,
        tools: list[BaseNode] | None = None,
        memory_store: MemoryStore | None = None,
        cache: CacheBackend | None = None,
        cache_ttl: int | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tool_iterations: int = 5,
        repair_attempts: int = 1,
        name: str = "agent",
    ) -> None:
        # Compose: tools -> memory -> cache -> raw provider.
        wrapped: BaseLLMNode = llm
        if cache is not None:
            wrapped = CachedLLMNode(name=f"{name}-cache", inner=wrapped, cache=cache, ttl=cache_ttl)
        if memory_store is not None:
            wrapped = MemoryNode(name=f"{name}-memory", inner=wrapped, store=memory_store)

        wrapped.max_tool_iterations = max_tool_iterations
        if tools:
            wrapped.bind_tools(tools)

        self._llm = wrapped
        self.model = model
        self.system_prompt = system_prompt
        self.response_schema = response_schema
        self.repair_attempts = repair_attempts
        self.name = name

    def _build_messages(self, user_message: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize the user-supplied input to an OpenAI-style messages list."""
        messages: list[dict[str, Any]] = []
        if isinstance(user_message, list):
            # Caller passed a full message history; respect it but still
            # prepend system prompt if one is set and not already present.
            messages = list(user_message)
            if self.system_prompt and not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            return messages

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _drive(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> tuple[LLMResponse, LLMUsage]:
        """Run the LLM driver loop and aggregate token usage across all
        provider round-trips. Returns the last LLMResponse and the
        cumulative LLMUsage."""
        last: LLMResponse | None = None
        total = LLMUsage()
        async for response in self._llm.async_call(
            model=self.model, messages=messages, params=params
        ):
            last = response
            total.prompt_tokens += response.usage.prompt_tokens
            total.completion_tokens += response.usage.completion_tokens
            total.total_tokens += response.usage.total_tokens
        if last is None:
            raise RuntimeError(
                f"{self.name}: LLM driver yielded zero responses (provider misbehaved)"
            )
        return last, total

    async def run(
        self,
        user_message: str | list[dict[str, Any]],
        *,
        thread_id: str = "default",
        params: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run one agent turn and return a structured :class:`AgentResult`.

        Args:
            user_message: Either a plain string (wrapped as a user
                message) or a full OpenAI-style messages list.
            thread_id: Conversation thread id when memory is attached.
                Ignored if no memory store was configured.
            params: Additional per-call provider knobs forwarded to
                ``_complete`` (e.g. ``{"temperature": 0.2}``).

        Returns:
            :class:`AgentResult` with the terminal content, optional
            parsed schema instance, cumulative token usage, and the
            raw final :class:`LLMResponse`.
        """
        base_params: dict[str, Any] = {"thread_id": thread_id, **(params or {})}
        messages = self._build_messages(user_message)

        if self.response_schema is None:
            last, total = await self._drive(messages, base_params)
            return AgentResult(
                content=last.content,
                tool_calls=last.tool_calls,
                parsed=None,
                usage=total,
                raw=last,
            )

        # Schema-validated path: drive the LLM, parse JSON, repair on
        # failure. The repair loop mirrors call_with_schema's, but here
        # we have the LLMResponse handy so we can return it (and the
        # cumulative usage) alongside the parsed model.
        attempts = self.repair_attempts + 1
        last_response: LLMResponse | None = None
        last_error: Exception | None = None
        last_content = ""
        cumulative = LLMUsage()

        for attempt in range(attempts):
            last_response, turn_usage = await self._drive(messages, base_params)
            cumulative.prompt_tokens += turn_usage.prompt_tokens
            cumulative.completion_tokens += turn_usage.completion_tokens
            cumulative.total_tokens += turn_usage.total_tokens
            last_content = last_response.content

            try:
                payload = _extract_json_payload(last_content)
                data = json.loads(payload)
                parsed = self.response_schema.model_validate(data)
                return AgentResult(
                    content=last_content,
                    tool_calls=last_response.tool_calls,
                    parsed=parsed,
                    usage=cumulative,
                    raw=last_response,
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                messages.append({"role": "assistant", "content": last_content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not parse as the expected schema. "
                            f"Error: {exc}\n\nReturn ONLY a JSON object matching this schema "
                            "(no prose, no fences):\n"
                            f"{json.dumps(self.response_schema.model_json_schema())}"
                        ),
                    }
                )

        raise StructuredOutputError(
            f"{self.name}: LLM output did not match schema after {attempts} attempt(s): {last_error}",
            last_content=last_content,
            last_error=last_error or RuntimeError("unknown"),
        )

    async def clear_history(self, thread_id: str = "default") -> None:
        """Drop the persisted history for ``thread_id``. No-op if the
        agent wasn't constructed with a memory store."""
        node: Any = self._llm
        while node is not None:
            if isinstance(node, MemoryNode):
                await node.store.clear(thread_id)
                return
            node = getattr(node, "inner", None)
