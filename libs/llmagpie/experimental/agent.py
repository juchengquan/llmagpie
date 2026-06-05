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
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmagpie.base.node import BaseNode
from llmagpie.observability import (
    RunContext,
    agent_span,
    attach_context,
    capture_to,
    derive,
    push,
    resolve_debug_path,
)

from .nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from .nodes.generators.cache import CacheBackend, CachedLLMNode
from .nodes.generators.memory import MemoryNode, MemoryStore
from .nodes.generators.structured import StructuredOutputError, _extract_json_payload

M = TypeVar("M", bound=BaseModel)


class BudgetExceededError(RuntimeError):
    """Raised by :class:`Agent` when a single ``run()`` would exceed
    the caller-supplied token or cost ceiling. ``usage_so_far`` reflects
    the cumulative consumption up to (and including) the last
    completed provider call before the budget was tripped.

    When raised inside a framework entry point, ``run_context`` is
    populated by :func:`llmagpie.observability.attach_context` so the
    post-mortem helper can render the in-flight delegation trace.
    """

    def __init__(
        self,
        message: str,
        *,
        usage_so_far: LLMUsage,
        budget_limit: float,
        budget_dimension: str,
    ) -> None:
        super().__init__(message)
        self.usage_so_far = usage_so_far
        self.budget_limit = budget_limit
        self.budget_dimension = budget_dimension
        # Populated by attach_context() when the error bubbles through
        # an Agent/Supervisor entry point. None outside of a run.
        self.run_context: RunContext | None = None


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
    # Snapshot of the RunContext that owned the run. Lets callers
    # correlate the result back to logs / traces / debug tapes by
    # `run_context.run_id` without holding a ContextVar reference.
    run_context: RunContext | None = None
    # Path to the JSONL tape written when the agent ran with
    # ``debug=True``. ``None`` for the default (no-capture) path.
    tape_path: Path | None = None


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
        max_tokens_per_run: int | None = None,
        max_cost_per_run: float | None = None,
        cost_per_1k_tokens: dict[str, float] | None = None,
        stop_condition: Any = None,
        debug: bool = False,
        debug_dir: str | Path | None = None,
        name: str = "agent",
    ) -> None:
        # Compose: tools -> memory -> cache -> raw provider.
        wrapped: BaseLLMNode = llm
        if cache is not None:
            wrapped = CachedLLMNode(name=f"{name}-cache", inner=wrapped, cache=cache, ttl=cache_ttl)
        if memory_store is not None:
            wrapped = MemoryNode(name=f"{name}-memory", inner=wrapped, store=memory_store)

        wrapped.max_tool_iterations = max_tool_iterations
        if stop_condition is not None:
            wrapped.stop_condition = stop_condition
        if tools:
            wrapped.bind_tools(tools)

        self._llm = wrapped
        self.model = model
        self.system_prompt = system_prompt
        self.response_schema = response_schema
        self.repair_attempts = repair_attempts
        self.max_tokens_per_run = max_tokens_per_run
        self.max_cost_per_run = max_cost_per_run
        # Keys: {"prompt", "completion"} → $/1k tokens. Optional; only
        # consulted when `max_cost_per_run` is set or when callers
        # request `cost_of(usage)` directly.
        self.cost_per_1k_tokens = cost_per_1k_tokens or {}
        self.name = name
        # Debug-mode capture: when True, ``run()`` opens a ``capture_to``
        # context so every LLM round-trip lands in a per-run JSONL tape
        # under ``debug_dir`` (defaults to ``./.llmagpie-debug/``).
        self.debug = debug
        self.debug_dir = debug_dir

    def cost_of(self, usage: LLMUsage) -> float:
        """Convert an :class:`LLMUsage` to a dollar figure using the
        agent's ``cost_per_1k_tokens`` table. Returns 0.0 if no price
        table was configured."""
        if not self.cost_per_1k_tokens:
            return 0.0
        prompt_price = self.cost_per_1k_tokens.get("prompt", 0.0)
        completion_price = self.cost_per_1k_tokens.get("completion", 0.0)
        return (
            usage.prompt_tokens / 1000.0 * prompt_price
            + usage.completion_tokens / 1000.0 * completion_price
        )

    def _enforce_budget(self, usage: LLMUsage) -> None:
        """Check the cumulative ``usage`` against the configured
        ceilings. Raises :class:`BudgetExceededError` on violation."""
        if self.max_tokens_per_run is not None and usage.total_tokens > self.max_tokens_per_run:
            raise BudgetExceededError(
                f"{self.name}: run exceeded max_tokens_per_run "
                f"({usage.total_tokens} > {self.max_tokens_per_run})",
                usage_so_far=usage,
                budget_limit=float(self.max_tokens_per_run),
                budget_dimension="tokens",
            )
        if self.max_cost_per_run is not None:
            cost = self.cost_of(usage)
            if cost > self.max_cost_per_run:
                raise BudgetExceededError(
                    f"{self.name}: run exceeded max_cost_per_run "
                    f"(${cost:.4f} > ${self.max_cost_per_run:.4f})",
                    usage_so_far=usage,
                    budget_limit=self.max_cost_per_run,
                    budget_dimension="cost",
                )

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
            # Enforce per-run budget after each provider round-trip so
            # tool-call loops can't silently overspend.
            self._enforce_budget(total)
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

        # Enter a fresh ``RunContext`` (or a child of an outer one when
        # this Agent is nested inside a Supervisor / WorkerHandle).
        # ``derive`` inherits ``run_id`` / ``supervisor`` / ``depth``
        # from the parent and overrides only what this frame owns.
        ctx = derive(agent=self.name, thread_id=thread_id)
        capture_cm = self._make_capture_cm(ctx)
        with push(ctx), agent_span(agent_name=self.name), capture_cm as tape:
            tape_path = tape.path if tape is not None else None
            try:
                if self.response_schema is None:
                    last, total = await self._drive(messages, base_params)
                    return AgentResult(
                        content=last.content,
                        tool_calls=last.tool_calls,
                        parsed=None,
                        usage=total,
                        raw=last,
                        run_context=ctx,
                        tape_path=tape_path,
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
                            run_context=ctx,
                            tape_path=tape_path,
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
            except Exception as exc:
                attach_context(exc, ctx)
                raise

    def _make_capture_cm(self, ctx: RunContext) -> Any:
        """Build the capture context for this run. Returns a real
        :func:`capture_to` ctx when ``debug=True``, else a null ctx
        that yields ``None`` so callers can branch on the value."""
        if not self.debug:
            return nullcontext(None)
        path = resolve_debug_path(
            debug_dir=self.debug_dir, run_id=ctx.run_id, agent_label=self.name
        )
        return capture_to(path, agent_label=self.name)

    async def stream(
        self,
        user_message: str | list[dict[str, Any]],
        *,
        thread_id: str = "default",
        params: dict[str, Any] | None = None,
    ):
        """Stream a single LLM round-trip as :class:`StreamChunk` deltas.

        Unlike :meth:`run`, ``stream`` does NOT drive the tool-call
        loop — it yields chunks from one provider call. Once the
        stream completes, callers can inspect the assembled response
        and dispatch tools manually if needed (or just use
        :meth:`run` instead for non-streaming tool agents).

        Memory and cache short-circuits behave the same as :meth:`run`:
        the assembled final response is appended to per-thread memory
        on stream completion.

        Args:
            user_message: Plain string or full messages list.
            thread_id: Conversation thread when memory is attached.
            params: Forwarded to ``stream_complete`` as kwargs.

        Yields:
            :class:`StreamChunk` objects from the underlying provider.

        Raises:
            NotImplementedError: if the configured provider doesn't
                implement ``stream_complete``.
        """
        from .nodes.generators._base import StreamChunk

        base_params: dict[str, Any] = {"thread_id": thread_id, **(params or {})}
        messages = self._build_messages(user_message)

        # Walk down through memory/cache wrappers to find the bottom
        # provider that actually streams. Memory and cache currently
        # don't have streaming variants; for v1 we bypass them on the
        # stream path and (if memory is attached) write the assembled
        # final response back ourselves so history still accumulates.
        innermost: Any = self._llm
        while hasattr(innermost, "inner") and innermost.inner is not None:
            innermost = innermost.inner

        # Pop thread_id; the inner provider doesn't know about it.
        kwargs = {k: v for k, v in base_params.items() if k != "thread_id"}

        chunks: list[StreamChunk] = []
        async for chunk in innermost.stream_complete(self.model, messages, **kwargs):
            chunks.append(chunk)
            yield chunk

        # If memory is attached, persist the assembled exchange.
        async def _replay():
            for c in chunks:
                yield c

        if any(self._has_memory(w) for w in self._wrapper_chain()):
            final = await innermost.collect_stream(_replay())
            await self._persist_exchange(thread_id, messages, final)

    # --- helpers for stream() ---

    def _wrapper_chain(self) -> list[Any]:
        chain: list[Any] = []
        node: Any = self._llm
        while node is not None:
            chain.append(node)
            node = getattr(node, "inner", None)
        return chain

    def _has_memory(self, wrapper: Any) -> bool:
        return type(wrapper).__name__ == "MemoryNode"

    async def _persist_exchange(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        response: LLMResponse,
    ) -> None:
        """Mirror MemoryNode._complete's persistence step for the
        stream path. We append the caller's new messages plus the
        assistant turn, but not the historical messages already in
        the store."""
        for wrapper in self._wrapper_chain():
            if self._has_memory(wrapper):
                turn: dict[str, Any] = {
                    "role": response.role,
                    "content": response.content,
                }
                if response.tool_calls:
                    turn["tool_calls"] = response.tool_calls
                await wrapper.store.append(thread_id, [*messages, turn])
                return

    async def clear_history(self, thread_id: str = "default") -> None:
        """Drop the persisted history for ``thread_id``. No-op if the
        agent wasn't constructed with a memory store."""
        node: Any = self._llm
        while node is not None:
            if isinstance(node, MemoryNode):
                await node.store.clear(thread_id)
                return
            node = getattr(node, "inner", None)

    def as_worker(
        self,
        name: str,
        description: str,
        *,
        context_handoff: Any = "task_only",
        history_window: int = 6,
        persistent_thread: bool = False,
    ) -> Any:
        """Wrap this agent as a :class:`WorkerHandle` for use by a
        :class:`Supervisor`.

        ``name`` becomes the tool name the supervisor's LLM sees
        (prefixed with ``transfer_to_``). ``description`` should be
        action-oriented ("Use to gather authoritative sources on a
        topic.") so the supervisor's LLM knows when to delegate.

        See :class:`llmagpie.experimental.orchestration.WorkerHandle`
        for the full parameter reference.
        """
        # Lazy import to avoid a circular dependency (orchestration imports Agent).
        from .orchestration._worker import WorkerHandle

        return WorkerHandle(
            name=name,
            description=description,
            agent=self,
            context_handoff=context_handoff,
            history_window=history_window,
            persistent_thread=persistent_thread,
        )
