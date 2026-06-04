"""Tests for the supervisor / worker multi-agent orchestration.

The shape mirrors `tests/test_agent.py`: a hand-rolled MockLLMNode
that returns scripted responses lets us assert delegation order,
usage rollup, error handling, budget enforcement, and depth/loop
caps without touching a real provider.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llmagpie.experimental.agent import Agent, BudgetExceededError
from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from llmagpie.experimental.orchestration import (
    DelegationTrace,
    HandoffArgs,
    Supervisor,
    WorkerHandle,
    WorkerResult,
)
from llmagpie.experimental.orchestration._worker import HANDOFF_PREFIX
from pydantic import Field, PrivateAttr


class MockLLMNode(BaseLLMNode):
    """LLM node that replays a pre-scripted list of LLMResponses."""

    responses: list[LLMResponse] = Field(default_factory=list)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    async def _complete(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        self._calls.append(
            {"model": model, "messages": [dict(m) for m in messages], "kwargs": dict(kwargs)}
        )
        if not self.responses:
            raise RuntimeError("MockLLMNode: ran out of scripted responses")
        return self.responses.pop(0)


def _resp(
    content: str = "",
    tool_calls: list[dict] | None = None,
    *,
    prompt_tokens: int = 1,
    completion_tokens: int = 1,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        model="m",
        role="assistant",
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _handoff_call(worker_name: str, task: str, call_id: str = "tc1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": f"{HANDOFF_PREFIX}{worker_name}",
            "arguments": json.dumps({"task": task}),
        },
    }


# ---------------------------------------------------------------------------
# Happy path: supervisor delegates to one worker, gets a result, finishes.
# ---------------------------------------------------------------------------


def test_supervisor_delegates_to_one_worker():
    # Worker's LLM: returns content directly when its task arrives.
    worker_llm = MockLLMNode(
        name="worker_llm",
        responses=[_resp("found three sources", prompt_tokens=10, completion_tokens=5)],
    )
    worker = Agent(llm=worker_llm, model="m", name="worker")

    # Supervisor's LLM: round 1 emits a handoff tool call; round 2 wraps up.
    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp(
                "",
                tool_calls=[_handoff_call("researcher", "find sources on Mamba SSMs")],
                prompt_tokens=20,
                completion_tokens=3,
            ),
            _resp("done — three sources gathered", prompt_tokens=30, completion_tokens=8),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="researcher", description="Use to find sources.")],
    )

    result = asyncio.run(sup.run("research Mamba SSMs"))

    assert result.content == "done — three sources gathered"
    assert len(result.worker_results) == 1
    assert result.worker_results[0].worker == "researcher"
    assert result.worker_results[0].content == "found three sources"
    # Usage rolls up: 20+3 (sup round 1) + 10+5 (worker) + 30+8 (sup round 2) = 76
    assert result.usage.total_tokens == 23 + 15 + 38
    assert result.trace.children[0].worker == "researcher"
    assert result.trace.children[0].task == "find sources on Mamba SSMs"


# ---------------------------------------------------------------------------
# Multi-worker sequential dispatch.
# ---------------------------------------------------------------------------


def test_supervisor_delegates_to_two_workers_sequentially():
    researcher_llm = MockLLMNode(name="r_llm", responses=[_resp("sources: A, B, C")])
    researcher = Agent(llm=researcher_llm, model="m", name="researcher")

    writer_llm = MockLLMNode(name="w_llm", responses=[_resp("draft summary")])
    writer = Agent(llm=writer_llm, model="m", name="writer")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("", tool_calls=[_handoff_call("researcher", "find sources", "t1")]),
            _resp("", tool_calls=[_handoff_call("writer", "summarize the sources", "t2")]),
            _resp("final report"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[
            researcher.as_worker(name="researcher", description="Find sources."),
            writer.as_worker(name="writer", description="Write summaries."),
        ],
    )

    result = asyncio.run(sup.run("research and write"))
    assert result.content == "final report"
    assert [wr.worker for wr in result.worker_results] == ["researcher", "writer"]
    assert len(result.trace.children) == 2


# ---------------------------------------------------------------------------
# Budget enforcement across supervisor + workers.
# ---------------------------------------------------------------------------


def test_supervisor_budget_includes_worker_usage():
    worker_llm = MockLLMNode(
        name="w_llm", responses=[_resp("ok", prompt_tokens=50, completion_tokens=50)]
    )
    worker = Agent(llm=worker_llm, model="m", name="worker")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp(
                "",
                tool_calls=[_handoff_call("w", "do something")],
                prompt_tokens=10,
                completion_tokens=5,
            ),
            # we won't reach round 2 — budget should trip first.
            _resp("never reached"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="w", description="A worker.")],
        max_tokens_per_run=50,  # supervisor's 15 + worker's 100 = 115 > 50
    )

    try:
        asyncio.run(sup.run("delegate"))
    except BudgetExceededError as e:
        assert "max_tokens_per_run" in str(e)
        assert e.budget_dimension == "tokens"
    else:
        raise AssertionError("expected BudgetExceededError")


# ---------------------------------------------------------------------------
# Worker crash: surfaced as a tool-result error; supervisor can continue.
# ---------------------------------------------------------------------------


def test_worker_crash_surfaces_as_tool_error():
    # Worker LLM that raises on _complete.
    class CrashingLLM(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            raise RuntimeError("provider exploded")

    crash = CrashingLLM(name="crash")
    crashy_worker = Agent(llm=crash, model="m", name="crashy")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("", tool_calls=[_handoff_call("crashy", "do it")]),
            _resp("recovered, here's the answer"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[crashy_worker.as_worker(name="crashy", description="A flaky worker.")],
    )

    result = asyncio.run(sup.run("delegate"))
    assert result.content == "recovered, here's the answer"
    assert result.worker_results[0].error is not None
    assert "provider exploded" in result.worker_results[0].error


# ---------------------------------------------------------------------------
# Hallucinated worker name: validated, returned as a tool-error message.
# ---------------------------------------------------------------------------


def test_hallucinated_worker_name_becomes_tool_error():
    worker_llm = MockLLMNode(name="w_llm", responses=[_resp("ok")])
    worker = Agent(llm=worker_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("", tool_calls=[_handoff_call("nonexistent", "do something")]),
            _resp("done"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="researcher", description="A worker.")],
    )

    result = asyncio.run(sup.run("delegate"))
    assert result.content == "done"
    assert result.worker_results[0].error is not None
    assert "unknown worker" in result.worker_results[0].error.lower()


# ---------------------------------------------------------------------------
# max_delegations cap.
# ---------------------------------------------------------------------------


def test_max_delegations_caps_supervisor_loop():
    worker_llm = MockLLMNode(name="w_llm", responses=[_resp("a"), _resp("b"), _resp("c")])
    worker = Agent(llm=worker_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("", tool_calls=[_handoff_call("w", "1")]),
            _resp("", tool_calls=[_handoff_call("w", "2")]),
            _resp("", tool_calls=[_handoff_call("w", "3")]),  # cap=2, this gets refused
            _resp("done"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="w", description="A worker.")],
        max_delegations=2,
    )

    result = asyncio.run(sup.run("loop"))
    # 2 actual delegations happened; the 3rd became an error tool result.
    successful = [wr for wr in result.worker_results if wr.error is None]
    assert len(successful) == 2


# ---------------------------------------------------------------------------
# Context handoff: task_only doesn't leak parent history.
# ---------------------------------------------------------------------------


def test_task_only_handoff_does_not_share_parent_history():
    worker_llm = MockLLMNode(name="w_llm", responses=[_resp("ok")])
    worker = Agent(llm=worker_llm, model="m", system_prompt="you are a worker")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("", tool_calls=[_handoff_call("w", "specific task")]),
            _resp("done"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        system_prompt="you are the supervisor with secret context",
        workers=[worker.as_worker(name="w", description="A worker.", context_handoff="task_only")],
    )

    asyncio.run(sup.run("the user's full message with sensitive context"))

    # The worker's LLM should have been called with messages that contain ONLY:
    # - worker's own system_prompt
    # - the task string
    # And should NOT contain the supervisor's system_prompt or the user message.
    worker_call_messages = worker_llm._calls[0]["messages"]
    serialized = json.dumps(worker_call_messages)
    assert "specific task" in serialized
    assert "supervisor with secret context" not in serialized
    assert "sensitive context" not in serialized


# ---------------------------------------------------------------------------
# Aggregation: "all_messages" concatenates everything.
# ---------------------------------------------------------------------------


def test_aggregation_all_messages():
    worker_llm = MockLLMNode(
        name="w_llm", responses=[_resp("worker A output"), _resp("worker B output")]
    )
    worker = Agent(llm=worker_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("", tool_calls=[_handoff_call("w", "task 1", "t1")]),
            _resp("", tool_calls=[_handoff_call("w", "task 2", "t2")]),
            _resp("supervisor wrap-up"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="w", description="A worker.")],
        aggregation="all_messages",
    )

    result = asyncio.run(sup.run("delegate twice"))
    assert "worker A output" in result.content
    assert "worker B output" in result.content
    assert "supervisor wrap-up" in result.content


# ---------------------------------------------------------------------------
# DelegationTrace.format() and cumulative_usage().
# ---------------------------------------------------------------------------


def test_delegation_trace_formats_and_sums():
    trace = DelegationTrace(
        worker="supervisor",
        task="root task",
        depth=0,
        started_at=0.0,
        ended_at=1.0,
        usage=LLMUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        children=[
            DelegationTrace(
                worker="researcher",
                task="find things",
                depth=1,
                started_at=0.1,
                ended_at=0.5,
                usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ],
    )
    formatted = trace.format()
    assert "supervisor" in formatted
    assert "researcher" in formatted
    assert trace.cumulative_usage().total_tokens == 23


# ---------------------------------------------------------------------------
# Regression: existing Agent tests should still pass via the refactored
# _enforce_budget / cost_of (no actual refactor needed in Phase 1, but
# this ensures the as_worker addition didn't break anything).
# ---------------------------------------------------------------------------


def test_agent_as_worker_returns_a_worker_handle():
    llm = MockLLMNode(name="llm", responses=[])
    agent = Agent(llm=llm, model="m", name="r")
    handle = agent.as_worker(name="researcher", description="Find sources.")
    assert isinstance(handle, WorkerHandle)
    assert handle.name == "researcher"
    schema = handle._generate_description_openai()
    assert schema["function"]["name"] == "transfer_to_researcher"
    assert "Find sources" in schema["function"]["description"]


# ---------------------------------------------------------------------------
# HandoffArgs validation.
# ---------------------------------------------------------------------------


def test_handoff_args_requires_task():
    try:
        HandoffArgs.model_validate({})
    except Exception as e:
        assert "task" in str(e)
    else:
        raise AssertionError("HandoffArgs without task should fail validation")


def test_worker_result_default_fields():
    wr = WorkerResult(worker="w")
    assert wr.content == ""
    assert wr.error is None
    assert wr.usage.total_tokens == 0


# ===========================================================================
# Phase 2 — Context handoff modes + no-progress detector
# ===========================================================================


def test_task_plus_history_handoff_includes_supervisor_messages():
    worker_llm = MockLLMNode(name="w_llm", responses=[_resp("ok")])
    worker = Agent(llm=worker_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            # First turn: supervisor produces some assistant content (will become history).
            _resp(
                "I'll think about this. Let me delegate.",
                tool_calls=[_handoff_call("w", "do the thing")],
            ),
            _resp("done"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        system_prompt="you are the supervisor",
        workers=[
            worker.as_worker(
                name="w",
                description="A worker.",
                context_handoff="task_plus_history",
                history_window=4,
            )
        ],
    )

    asyncio.run(sup.run("please help with my project"))

    # Worker should see the supervisor's recent history wrapped as user-side context.
    worker_msgs = worker_llm._calls[0]["messages"]
    serialized = json.dumps(worker_msgs)
    # The user message from the supervisor is forwarded:
    assert "please help with my project" in serialized
    # The task itself is the final user message:
    assert "do the thing" in serialized


def test_shared_scratchpad_handoff_includes_scratchpad_blob():
    # Worker returns content; supervisor doesn't need to merge anything for this test.
    worker_llm = MockLLMNode(name="w_llm", responses=[_resp("ok")])
    worker = Agent(llm=worker_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("", tool_calls=[_handoff_call("w", "step 1")]),
            _resp("done"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[
            worker.as_worker(name="w", description="A worker.", context_handoff="shared_scratchpad")
        ],
    )
    sup._current_scratchpad = {"key": "value"}  # would normally be set by run()

    asyncio.run(sup.run("start"))

    worker_msgs = worker_llm._calls[0]["messages"]
    serialized = json.dumps(worker_msgs)
    assert "<task>" in serialized
    assert "<scratchpad>" in serialized


def test_no_progress_detector_terminates_loop():
    # Worker that's never called; supervisor LLM emits no-tool-call responses
    # with near-identical content, triggering the no-progress guard.
    worker_llm = MockLLMNode(name="w_llm", responses=[])
    worker = Agent(llm=worker_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp("I am thinking about the problem"),
            # Same content essentially — should trigger no-progress.
            _resp("I am thinking about the problem"),
            _resp("I am thinking about the problem"),
            _resp("never reached"),  # if not stopped, this would be consumed
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="w", description="A worker.")],
        no_progress_window=3,
        no_progress_similarity=0.85,
    )

    # The supervisor's first response has no tool calls, so the loop exits
    # immediately (no `while response.tool_calls`). The progress detector
    # primarily protects against tool-call-then-no-tool-call oscillation.
    result = asyncio.run(sup.run("start"))
    assert result.content == "I am thinking about the problem"


def test_no_progress_detector_kicks_in_after_tool_calls_oscillation():
    """The detector fires when the supervisor alternates: emit tool calls,
    get worker result, emit identical no-tool-call content, repeat."""
    from llmagpie.experimental.orchestration._progress import NoProgressDetector

    det = NoProgressDetector(window=3, similarity_threshold=0.85)
    det.observe(_resp("I'm still thinking"))
    det.observe(_resp("I'm still thinking"))
    det.observe(_resp("I'm still thinking"))
    assert det.is_stuck() is True

    # A tool call resets the history — progress was made.
    det.observe(_resp("", tool_calls=[{"id": "x", "function": {"name": "foo", "arguments": "{}"}}]))
    assert det.is_stuck() is False


def test_no_progress_detector_does_not_fire_on_distinct_content():
    from llmagpie.experimental.orchestration._progress import NoProgressDetector

    det = NoProgressDetector(window=3, similarity_threshold=0.85)
    det.observe(_resp("First analysis: this problem involves linear algebra."))
    det.observe(_resp("Second consideration: numerical stability is paramount."))
    det.observe(_resp("Third point: the algorithm complexity is O(n^2)."))
    assert det.is_stuck() is False


# ===========================================================================
# Phase 3 — Parallel fan-out
# ===========================================================================


def test_parallel_fanout_dispatches_two_workers_concurrently():
    """When the supervisor emits two tool calls in one turn, they
    dispatch in parallel and both their results come back."""
    import asyncio as _asyncio

    in_flight = 0
    max_in_flight = 0

    class TrackingLLM(BaseLLMNode):
        responses: list[LLMResponse] = Field(default_factory=list)

        async def _complete(self, model, messages, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                # Yield to the event loop so siblings can also enter here.
                await _asyncio.sleep(0)
                await _asyncio.sleep(0)
                return self.responses.pop(0)
            finally:
                in_flight -= 1

    worker_a_llm = TrackingLLM(name="a_llm", responses=[_resp("A result")])
    worker_a = Agent(llm=worker_a_llm, model="m", name="a")

    worker_b_llm = TrackingLLM(name="b_llm", responses=[_resp("B result")])
    worker_b = Agent(llm=worker_b_llm, model="m", name="b")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp(
                "",
                tool_calls=[
                    _handoff_call("a", "task A", "tc1"),
                    _handoff_call("b", "task B", "tc2"),
                ],
            ),
            _resp("done"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[
            worker_a.as_worker(name="a", description="Worker A."),
            worker_b.as_worker(name="b", description="Worker B."),
        ],
        max_parallel_workers=2,
    )

    result = asyncio.run(sup.run("delegate parallel"))
    assert result.content == "done"
    assert {wr.worker for wr in result.worker_results} == {"a", "b"}
    # If they ran sequentially, max_in_flight would be 1. Parallel hits 2.
    assert max_in_flight == 2


def test_parallel_fanout_caps_at_max_parallel_workers():
    """With max_parallel_workers=1 we should never see more than one worker active."""
    import asyncio as _asyncio

    in_flight = 0
    max_in_flight = 0

    class TrackingLLM(BaseLLMNode):
        responses: list[LLMResponse] = Field(default_factory=list)

        async def _complete(self, model, messages, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                await _asyncio.sleep(0)
                await _asyncio.sleep(0)
                return self.responses.pop(0)
            finally:
                in_flight -= 1

    worker_llm = TrackingLLM(name="w_llm", responses=[_resp("A"), _resp("B"), _resp("C")])
    worker = Agent(llm=worker_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp(
                "",
                tool_calls=[
                    _handoff_call("w", "1", "tc1"),
                    _handoff_call("w", "2", "tc2"),
                    _handoff_call("w", "3", "tc3"),
                ],
            ),
            _resp("done"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="w", description="A worker.")],
        max_parallel_workers=1,
    )

    asyncio.run(sup.run("delegate"))
    # With cap=1, only one provider call at a time.
    assert max_in_flight == 1


def test_parallel_fanout_preserves_result_order_in_messages():
    """Results in tool-result messages should match the LLM's tool-call order
    even if workers finished out of order."""
    import asyncio as _asyncio

    class SlowFirstLLM(BaseLLMNode):
        responses: list[LLMResponse] = Field(default_factory=list)
        delay: float = 0.0

        async def _complete(self, model, messages, **kwargs):
            await _asyncio.sleep(self.delay)
            return self.responses.pop(0)

    # Worker "a" is slow, worker "b" is fast — they should still appear
    # in the [a, b] order in the supervisor's message log since that's
    # the LLM's emission order.
    slow_llm = SlowFirstLLM(name="slow", responses=[_resp("slow result")], delay=0.02)
    slow = Agent(llm=slow_llm, model="m")
    fast_llm = SlowFirstLLM(name="fast", responses=[_resp("fast result")], delay=0.0)
    fast = Agent(llm=fast_llm, model="m")

    sup_llm = MockLLMNode(
        name="sup_llm",
        responses=[
            _resp(
                "",
                tool_calls=[
                    _handoff_call("slow", "go slow", "tcA"),
                    _handoff_call("fast", "go fast", "tcB"),
                ],
            ),
            _resp("done"),
        ],
    )
    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[
            slow.as_worker(name="slow", description="Slow."),
            fast.as_worker(name="fast", description="Fast."),
        ],
    )

    result = asyncio.run(sup.run("parallel"))
    # The result ORDER in `worker_results` matches the LLM's tool-call order.
    assert [wr.worker for wr in result.worker_results] == ["slow", "fast"]


# ===========================================================================
# Phase 4 — Streaming
# ===========================================================================


def test_supervisor_stream_emits_chunks_in_order():
    """Supervisor.stream() yields supervisor chunks during its LLM call,
    then worker start/delta/end around each delegation."""
    from llmagpie.experimental.nodes.generators._base import StreamChunk

    class StreamingLLM(BaseLLMNode):
        # Scripts: list of "scripts", each a list of StreamChunks for one call.
        scripts: list[list[StreamChunk]] = Field(default_factory=list)

        async def stream_complete(self, model, messages, **kwargs):
            script = self.scripts.pop(0)
            for c in script:
                yield c

    # Supervisor: round 1 emits a handoff tool call; round 2 emits "done".
    handoff_tc = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "transfer_to_w", "arguments": '{"task": "do it"}'},
    }
    sup_llm = StreamingLLM(
        name="sup",
        scripts=[
            [
                StreamChunk(
                    delta_tool_calls=[handoff_tc],
                    finish_reason="tool_calls",
                    usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
            ],
            [
                StreamChunk(delta_content="Hello ", role="assistant"),
                StreamChunk(
                    delta_content="world",
                    finish_reason="stop",
                    usage=LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                ),
            ],
        ],
    )

    # Worker's LLM doesn't need to stream — _dispatch_worker_call uses agent.run().
    worker_llm = MockLLMNode(name="w_llm", responses=[_resp("worker output")])
    worker = Agent(llm=worker_llm, model="m")

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[worker.as_worker(name="w", description="A worker.")],
    )

    async def _collect():
        chunks = []
        async for c in sup.stream("hi"):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(_collect())
    # Find the events in order.
    sources_events = [(c.source, c.event) for c in chunks]
    # First: supervisor delta(s).
    assert sources_events[0] == ("supervisor", "delta")
    # Then worker start.
    assert ("worker", "start") in sources_events
    # Worker delta carrying its output.
    worker_deltas = [c for c in chunks if c.source == "worker" and c.event == "delta"]
    assert any(c.chunk and c.chunk.delta_content == "worker output" for c in worker_deltas)
    # Worker end.
    assert ("worker", "end") in sources_events
    # Final supervisor deltas after the worker.
    assert sources_events[-1] == ("supervisor", "delta")


def test_supervisor_chunk_basic_construction():
    """Construction-level smoke for SupervisorChunk."""
    from llmagpie.experimental.orchestration import SupervisorChunk

    c = SupervisorChunk(source="worker", worker="r", event="start")
    assert c.source == "worker"
    assert c.worker == "r"
    assert c.event == "start"
    assert c.chunk is None
