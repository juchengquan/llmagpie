"""Supervisor — an :class:`Agent` that delegates to worker agents.

Subclasses :class:`Agent` so it inherits ``cost_of`` / budget
enforcement / memory + cache wiring / structured outputs. Overrides
the driver loop (:meth:`_drive`) to intercept worker-handoff tool
calls and dispatch them async (workers are themselves agents, so
they need real async dispatch, not the ThreadPoolExecutor-backed
:class:`ToolsNode` path the supervisor still uses for regular
tools).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llmagpie.base.node import BaseNode
from llmagpie.observability import agent_span, attach_context, derive, push

from ..agent import Agent, AgentResult
from ..nodes.generators._base import LLMResponse, LLMUsage, StreamChunk
from ..nodes.generators.cache import CacheBackend
from ..nodes.generators.memory import MemoryStore
from ._progress import NoProgressDetector
from ._stream import SupervisorChunk
from ._trace import DelegationTrace
from ._worker import HANDOFF_PREFIX, HandoffArgs, WorkerHandle, WorkerResult

AggregationStrategy = Literal["last", "all_messages", "structured_merge"]


class SupervisorResult(AgentResult):
    """Result of a :meth:`Supervisor.run` call.

    Extends :class:`AgentResult` with the delegation trace and the
    list of every :class:`WorkerResult` produced during the run (in
    invocation order, so callers can correlate to ``trace.children``).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trace: DelegationTrace
    worker_results: list[WorkerResult] = Field(default_factory=list)


class Supervisor(Agent):
    """High-level supervisor that delegates tasks to worker agents.

    Args:
        llm, model, system_prompt, memory_store, cache, response_schema,
        max_tool_iterations, repair_attempts, max_tokens_per_run,
        max_cost_per_run, cost_per_1k_tokens, stop_condition, name:
            See :class:`Agent`.
        workers: List of :class:`WorkerHandle` registered with this
            supervisor. The supervisor's LLM sees them as tools named
            ``transfer_to_<worker.name>``.
        tools: Optional regular tools (non-worker) the supervisor can
            also call directly. Dispatched via the standard
            :class:`ToolsNode` path.
        max_delegations: Hard cap on total worker invocations per
            ``run()``. Default 10.
        max_depth: Cap on nesting depth when a worker is itself a
            supervisor. Default 3. Each level increments depth.
        aggregation: How to produce the final ``content`` after the
            supervisor's loop converges. ``"last"`` uses the
            supervisor's last assistant message (default). ``"all_messages"``
            concatenates every worker output. ``"structured_merge"``
            requires ``response_schema`` and routes through Agent's
            self-repair to produce a validated final. A callable
            ``(supervisor_messages, worker_results) -> str`` overrides
            entirely.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        *,
        workers: list[WorkerHandle],
        tools: list[BaseNode] | None = None,
        system_prompt: str | None = None,
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
        max_delegations: int = 10,
        max_depth: int = 3,
        aggregation: AggregationStrategy | Callable[[list[dict], list[WorkerResult]], str] = "last",
        no_progress_window: int = 3,
        no_progress_similarity: float = 0.85,
        max_parallel_workers: int = 4,
        debug: bool = False,
        debug_dir: Any = None,
        name: str = "supervisor",
    ) -> None:
        # Compose all tools so the LLM sees them in `_format_tools_for_provider`.
        # Workers are BaseNodes too, so they fit the same bind_tools path.
        all_tools: list[BaseNode] = list(tools or []) + list(workers)
        super().__init__(
            llm=llm,
            model=model,
            system_prompt=system_prompt,
            tools=all_tools if all_tools else None,
            memory_store=memory_store,
            cache=cache,
            cache_ttl=cache_ttl,
            response_schema=response_schema,
            max_tool_iterations=max_tool_iterations,
            repair_attempts=repair_attempts,
            max_tokens_per_run=max_tokens_per_run,
            max_cost_per_run=max_cost_per_run,
            cost_per_1k_tokens=cost_per_1k_tokens,
            stop_condition=stop_condition,
            debug=debug,
            debug_dir=debug_dir,
            name=name,
        )

        self.workers = list(workers)
        self._worker_map: dict[str, WorkerHandle] = {
            f"{HANDOFF_PREFIX}{w.name}": w for w in workers
        }
        # Names of the user's regular tools so we know which calls go through
        # ToolsNode vs. our async worker dispatch.
        self._regular_tool_names: set[str] = {t.name for t in (tools or [])}

        self.max_delegations = max_delegations
        self.max_depth = max_depth
        self.aggregation = aggregation
        self._stop_condition = stop_condition
        self._progress_window = no_progress_window
        self._progress_similarity = no_progress_similarity
        self._max_parallel_workers = max_parallel_workers

        # Per-run state populated by run(); reset on each entry.
        self._current_trace: DelegationTrace | None = None
        self._current_worker_results: list[WorkerResult] = []
        self._current_scratchpad: dict[str, Any] = {}
        self._current_depth: int = 0
        self._delegation_count: int = 0

    async def run(
        self,
        user_message: str | list[dict[str, Any]],
        *,
        thread_id: str = "default",
        params: dict[str, Any] | None = None,
        _depth: int = 0,
    ) -> SupervisorResult:
        """Run the supervisor loop. Returns a :class:`SupervisorResult`
        with the delegation trace and per-worker results alongside the
        usual :class:`AgentResult` fields.
        """
        # Per-run state.
        self._current_trace = DelegationTrace(
            worker=self.name,
            task=user_message if isinstance(user_message, str) else "<messages list>",
            depth=_depth,
            started_at=time.monotonic(),
        )
        self._current_worker_results = []
        self._current_scratchpad = {}
        self._current_depth = _depth
        self._delegation_count = 0

        # Push a RunContext for the supervisor so its workers (and any
        # downstream Agent.run() frames) inherit run_id / supervisor /
        # delegation_trace, and so any exception bubbles up carrying
        # the trace as it stood at failure.
        ctx = derive(
            agent=self.name,
            supervisor=self.name,
            depth=_depth,
            thread_id=thread_id,
            delegation_trace=self._current_trace,
        )
        capture_cm = self._make_capture_cm(ctx)
        with (
            push(ctx),
            agent_span(agent_name=self.name, is_supervisor=True),
            capture_cm as tape,
        ):
            tape_path = tape.path if tape is not None else None
            try:
                base_params: dict[str, Any] = {"thread_id": thread_id, **(params or {})}
                messages = self._build_messages(user_message)
                last, total, worker_results = await self._drive_supervisor(messages, base_params)
                self._current_trace.usage = total
                self._current_trace.ended_at = time.monotonic()

                final_content = self._finalize_content(messages, worker_results, last)

                return SupervisorResult(
                    content=final_content,
                    tool_calls=last.tool_calls,
                    parsed=None,  # structured_merge populates this elsewhere; see TODO
                    usage=total,
                    raw=last,
                    trace=self._current_trace,
                    worker_results=worker_results,
                    run_context=ctx,
                    tape_path=tape_path,
                )
            except Exception as exc:
                # Snapshot the in-flight trace state for the post-mortem
                # before re-raising. `attach_context` is idempotent — if
                # a deeper frame (e.g. a worker's Agent.run) already
                # stamped its own context, that one wins.
                if self._current_trace is not None:
                    self._current_trace.ended_at = time.monotonic()
                attach_context(exc, ctx)
                raise

    async def stream(
        self,
        user_message: str | list[dict[str, Any]],
        *,
        thread_id: str = "default",
        params: dict[str, Any] | None = None,
    ):
        """Stream a supervisor run as a sequence of :class:`SupervisorChunk` items.

        Yields three kinds of chunks:

        - ``source="supervisor"`` with a :class:`StreamChunk` — live tokens
          from the supervisor's own LLM call.
        - ``source="worker"`` with ``event="start"`` / ``event="end"`` —
          boundary markers around a worker invocation.
        - ``source="worker"`` with ``event="delta"`` and a :class:`StreamChunk`
          carrying the worker's final content — emitted once the worker
          completes (workers don't currently stream their own tool loops).

        The supervisor's loop terminates the same way as :meth:`run`
        (no tool calls + budget + max_delegations + no-progress).
        """
        # Reset per-run state — match run() semantics.
        self._current_trace = DelegationTrace(
            worker=self.name,
            task=user_message if isinstance(user_message, str) else "<messages list>",
            depth=0,
            started_at=time.monotonic(),
        )
        self._current_worker_results = []
        self._current_scratchpad = {}
        self._current_depth = 0
        self._delegation_count = 0

        messages = self._build_messages(user_message)
        stream_kwargs = {k: v for k, v in (params or {}).items() if k != "thread_id"}

        total = LLMUsage()
        iterations = 0
        max_iter = max(self.max_delegations + 5, 10)
        progress = NoProgressDetector(
            window=self._progress_window, similarity_threshold=self._progress_similarity
        )

        # Walk down to the innermost provider that can stream. Memory and cache
        # don't have streaming variants yet; we bypass them on the stream path,
        # same as Agent.stream() does.
        innermost: Any = self._llm
        while hasattr(innermost, "inner") and innermost.inner is not None:
            innermost = innermost.inner

        while True:
            chunks: list[StreamChunk] = []
            async for chunk in innermost.stream_complete(self.model, messages, **stream_kwargs):
                chunks.append(chunk)
                yield SupervisorChunk(source="supervisor", event="delta", chunk=chunk)

            # Assemble the supervisor's response from the stream.
            async def _replay(_chunks=chunks):
                for c in _chunks:
                    yield c

            response = await innermost.collect_stream(_replay())
            total = _accumulate(total, response.usage)
            self._enforce_budget(total)
            progress.observe(response)

            if not response.tool_calls or iterations >= max_iter:
                break
            if self._stop_condition is not None and self._stop_condition(response):
                break
            if progress.is_stuck():
                break

            # Append the assistant tool-call message.
            assistant_msg: dict[str, Any] = {
                "role": response.role or "assistant",
                "tool_calls": [
                    {
                        "id": tc.get("id"),
                        "type": tc.get("type", "function"),
                        "function": tc["function"],
                    }
                    for tc in response.tool_calls
                ],
            }
            if response.content:
                assistant_msg["content"] = response.content
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                name = tc.get("function", {}).get("name", "")
                if name not in self._worker_map and name not in self._regular_tool_names:
                    # Unknown tool — error message back to the LLM.
                    messages.append(self._tool_error_msg(tc, f"unknown tool: {name!r}"))
                    continue

                if name in self._worker_map:
                    worker = self._worker_map[name]
                    yield SupervisorChunk(source="worker", worker=worker.name, event="start")
                    if self._delegation_count >= self.max_delegations:
                        messages.append(self._tool_error_msg(tc, "max_delegations exceeded"))
                        yield SupervisorChunk(source="worker", worker=worker.name, event="end")
                        continue
                    wr = await self._dispatch_worker_call(tc, messages)
                    self._current_worker_results.append(wr)
                    total = _accumulate(total, wr.usage)
                    self._enforce_budget(total)
                    # Surface the worker's final content as a single delta chunk.
                    yield SupervisorChunk(
                        source="worker",
                        worker=worker.name,
                        event="delta",
                        chunk=StreamChunk(delta_content=wr.content, role="assistant"),
                    )
                    yield SupervisorChunk(source="worker", worker=worker.name, event="end")
                    messages.append(self._worker_result_msg(tc, wr))
                else:
                    # Regular (non-worker) tool — dispatch via ToolsNode.
                    tool_result = await self._llm.tools_node.async_call_(tool_calls_list=[tc])
                    tool_outputs = tool_result.get("tool_calls_list", [])
                    for ele in tool_outputs:
                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(ele.get("output"), default=str),
                                "tool_call_id": ele["id"],
                            }
                        )

            iterations += 1

        self._current_trace.usage = total
        self._current_trace.ended_at = time.monotonic()

    async def _drive_supervisor(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> tuple[LLMResponse, LLMUsage, list[WorkerResult]]:
        """The supervisor's main loop.

        Calls ``self._llm._complete`` directly (bypassing the LLM
        node's built-in tool-call loop) so we can route worker calls
        to our async dispatch while still letting regular tool calls
        go through the standard :class:`ToolsNode` path.
        """
        total = LLMUsage()
        worker_results: list[WorkerResult] = []
        iterations = 0
        max_iter = max(self.max_delegations + 5, 10)  # supervisor's own LLM rounds
        progress = NoProgressDetector(
            window=self._progress_window, similarity_threshold=self._progress_similarity
        )

        kwargs = {k: v for k, v in params.items() if k != "thread_id"}
        # MemoryNode is the outermost wrapper when memory is attached, and it
        # reads `thread_id` from kwargs to persist. Re-attach so memory still works.
        if any(_is_memory_wrapper(w) for w in self._wrapper_chain()):
            kwargs["thread_id"] = params.get("thread_id", "default")

        # First LLM call goes through self._llm._complete so memory/cache/provider
        # composition all apply.
        response = await self._llm._complete_traced(self.model, messages, **kwargs)
        total = _accumulate(total, response.usage)
        self._enforce_budget(total)
        progress.observe(response)

        if self._stop_condition is not None and self._stop_condition(response):
            return response, total, worker_results
        if progress.is_stuck():
            return response, total, worker_results

        while response.tool_calls and iterations < max_iter:
            # Separate worker calls from regular tool calls.
            worker_calls = []
            regular_calls = []
            for tc in response.tool_calls:
                name = tc.get("function", {}).get("name", "")
                if name in self._worker_map:
                    worker_calls.append(tc)
                elif name in self._regular_tool_names:
                    regular_calls.append(tc)
                else:
                    # Hallucinated tool — synthesize an error result the LLM can see.
                    worker_calls.append(tc)  # handled by _dispatch_worker_calls' fallback

            # Build the assistant message describing the tool calls.
            assistant_msg = {
                "role": response.role or "assistant",
                "tool_calls": [
                    {
                        "id": tc.get("id"),
                        "type": tc.get("type", "function"),
                        "function": tc["function"],
                    }
                    for tc in response.tool_calls
                ],
            }
            if response.content:
                assistant_msg["content"] = response.content
            messages.append(assistant_msg)

            # Dispatch worker calls: serial when only one, parallel when multiple.
            # Parallel uses asyncio.TaskGroup (structured concurrency) so a
            # sibling's failure cooperatively cancels its still-running peers.
            #
            # We pre-budget delegations: if we have more worker calls than the
            # remaining `max_delegations` cap allows, the surplus get refused
            # before being scheduled. Same for over-budget supervisors.
            allowed: list[dict] = []
            for tc in worker_calls:
                if self._delegation_count >= self.max_delegations:
                    messages.append(self._tool_error_msg(tc, "max_delegations exceeded"))
                    continue
                self._delegation_count += 1
                allowed.append(tc)

            if len(allowed) <= 1:
                for tc in allowed:
                    wr = await self._dispatch_worker_call(tc, messages, _count_already=True)
                    worker_results.append(wr)
                    total = _accumulate(total, wr.usage)
                    self._enforce_budget(total)
                    messages.append(self._worker_result_msg(tc, wr))
            else:
                results_by_call_id = await self._dispatch_workers_parallel(
                    allowed,
                    messages,
                )
                # results_by_call_id preserves the call order from `allowed`.
                for tc in allowed:
                    wr = results_by_call_id[tc.get("id", "")]
                    worker_results.append(wr)
                    total = _accumulate(total, wr.usage)
                    messages.append(self._worker_result_msg(tc, wr))
                # Budget check after all sibling results are tallied.
                self._enforce_budget(total)

            # Dispatch regular tools via the standard ToolsNode.
            if regular_calls and self._llm.tools_node is not None:
                tool_result = await self._llm.tools_node.async_call_(tool_calls_list=regular_calls)
                tool_outputs = tool_result.get("tool_calls_list", [])
                for ele in tool_outputs:
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(ele.get("output"), default=str),
                            "tool_call_id": ele["id"],
                        }
                    )

            # Next LLM round.
            response = await self._llm._complete_traced(self.model, messages, **kwargs)
            total = _accumulate(total, response.usage)
            self._enforce_budget(total)
            progress.observe(response)
            iterations += 1
            if self._stop_condition is not None and self._stop_condition(response):
                break
            if progress.is_stuck():
                break

        return response, total, worker_results

    async def _dispatch_workers_parallel(
        self,
        tcs: list[dict[str, Any]],
        parent_messages: list[dict[str, Any]],
    ) -> dict[str, WorkerResult]:
        """Run multiple worker calls concurrently under an
        :class:`asyncio.TaskGroup`. Bounded by
        ``max_parallel_workers`` via a semaphore.

        Each task returns a :class:`WorkerResult` (never raises — the
        worker handle catches its own exceptions). Returns a dict
        keyed by tool-call id so the caller can re-order outputs to
        match the LLM's tool-call sequence.
        """
        sem = asyncio.Semaphore(self._max_parallel_workers)
        results: dict[str, WorkerResult] = {}

        async def _one(tc: dict[str, Any]) -> None:
            async with sem:
                wr = await self._dispatch_worker_call(tc, parent_messages, _count_already=True)
                results[tc.get("id", "")] = wr

        async with asyncio.TaskGroup() as tg:
            for tc in tcs:
                tg.create_task(_one(tc))

        return results

    async def _dispatch_worker_call(
        self,
        tc: dict[str, Any],
        parent_messages: list[dict[str, Any]],
        *,
        _count_already: bool = False,
    ) -> WorkerResult:
        """Validate args, look up the worker, invoke it, record a trace
        node. Errors become :attr:`WorkerResult.error` — never raised
        to the caller.

        ``_count_already`` is set by callers that already incremented
        ``self._delegation_count`` before scheduling (parallel
        dispatch needs to pre-budget so we don't over-schedule)."""
        fn = tc.get("function", {})
        name = fn.get("name", "")

        if name not in self._worker_map:
            return WorkerResult(
                worker=name or "<unknown>",
                error=(f"unknown worker: {name!r}. Available: {sorted(self._worker_map.keys())}"),
            )

        # Parse args.
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args_dict = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError as exc:
                return WorkerResult(worker=name, error=f"malformed JSON args: {exc}")
        else:
            args_dict = raw_args
        try:
            args = HandoffArgs.model_validate(args_dict)
        except Exception as exc:
            return WorkerResult(worker=name, error=f"invalid handoff args: {exc}")

        worker = self._worker_map[name]
        new_depth = self._current_depth + 1

        if new_depth > self.max_depth:
            return WorkerResult(
                worker=worker.name,
                error=f"max_depth exceeded ({new_depth} > {self.max_depth})",
            )

        if not _count_already:
            self._delegation_count += 1

        # Trace node for this call.
        child = DelegationTrace(
            worker=worker.name,
            task=args.task,
            depth=new_depth,
            started_at=time.monotonic(),
        )
        if self._current_trace is not None:
            self._current_trace.children.append(child)

        # Dispatch. WorkerHandle.dispatch never raises — errors become WorkerResult.error.
        result = await worker.dispatch(
            task=args.task,
            context_hint=args.context_hint,
            parent_messages=parent_messages,
            depth=new_depth,
            scratchpad=self._current_scratchpad,
        )

        # Record back into the trace.
        child.usage = result.usage
        child.ended_at = time.monotonic()
        if result.error:
            child.error = result.error

        # If this worker uses shared_scratchpad and returned a structured
        # patch under `updates`, merge it.
        if worker.context_handoff == "shared_scratchpad" and result.parsed:
            updates = (
                result.parsed.get("updates")
                if isinstance(result.parsed, dict)
                else getattr(result.parsed, "updates", None)
            )
            if isinstance(updates, dict):
                self._current_scratchpad.update(updates)

        return result

    @staticmethod
    def _worker_result_msg(tc: dict[str, Any], result: WorkerResult) -> dict[str, Any]:
        """Wrap a :class:`WorkerResult` into the tool-result message the
        supervisor's LLM will see on the next turn."""
        payload: dict[str, Any] = {"worker": result.worker, "result": result.content}
        if result.parsed is not None:
            try:
                payload["structured"] = (
                    result.parsed.model_dump()
                    if isinstance(result.parsed, BaseModel)
                    else result.parsed
                )
            except Exception as exc:
                # Best-effort serialization: if the worker's parsed payload isn't
                # JSON-serializable (unusual custom types), omit the structured
                # field rather than failing the supervisor's round-trip. The raw
                # text content is always present on `result.content`.
                payload["structured_error"] = repr(exc)
        if result.error:
            payload["error"] = result.error
        return {
            "role": "tool",
            "tool_call_id": tc.get("id"),
            "content": json.dumps(payload, default=str),
        }

    @staticmethod
    def _tool_error_msg(tc: dict[str, Any], err: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tc.get("id"),
            "content": json.dumps({"error": err}),
        }

    def _finalize_content(
        self,
        messages: list[dict[str, Any]],
        worker_results: list[WorkerResult],
        last: LLMResponse,
    ) -> str:
        """Build the supervisor's final ``content`` per ``self.aggregation``."""
        agg = self.aggregation
        if callable(agg):
            return agg(messages, worker_results)
        if agg == "last":
            return last.content
        if agg == "all_messages":
            lines = []
            for wr in worker_results:
                if wr.error:
                    lines.append(f"[{wr.worker} ERROR] {wr.error}")
                else:
                    lines.append(f"[{wr.worker}] {wr.content}")
            if last.content:
                lines.append(f"[supervisor] {last.content}")
            return "\n\n".join(lines)
        if agg == "structured_merge":
            # The supervisor's response_schema (if set) applies via the
            # inherited Agent path. For Phase 1 we behave like "last" if
            # no schema is set, since true structured merge is a Phase-3
            # follow-up.
            return last.content
        raise ValueError(f"Unknown aggregation: {agg!r}")


def _accumulate(total: LLMUsage, delta: LLMUsage) -> LLMUsage:
    total.prompt_tokens += delta.prompt_tokens
    total.completion_tokens += delta.completion_tokens
    total.total_tokens += delta.total_tokens
    return total


def _is_memory_wrapper(node: Any) -> bool:
    return type(node).__name__ == "MemoryNode"
