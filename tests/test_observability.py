"""Tests for the observability primitives (Phase 1).

Covers:
- ``RunContext`` ContextVar propagation across ``await`` and across
  :class:`ThreadPoolExecutor` worker threads (via ``ToolsNode.fire``).
- ``derive`` inheritance semantics — child contexts inherit ``run_id``
  and parent fields, overriding only what they own.
- Exception enrichment: framework exceptions raised inside ``Agent.run``
  and ``Supervisor.run`` carry a ``run_context`` attribute with the
  in-flight delegation trace.
- ``format_error`` output: header line, context block, and the
  delegation-trace tree all render.
- Logging filter: the default logger gets ``run_id`` / ``agent`` /
  ``worker`` placeholders, and the values match the current context.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from llmagpie.base.logging import get_or_create_logger
from llmagpie.experimental.agent import Agent, BudgetExceededError
from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from llmagpie.experimental.orchestration import Supervisor
from llmagpie.observability import (
    RunContext,
    attach_context,
    current_context,
    derive,
    format_error,
    format_trace,
    push,
)
from pydantic import Field, PrivateAttr


class MockLLMNode(BaseLLMNode):
    """LLM that replays scripted responses; records the active
    ``RunContext.run_id`` at each call so tests can assert
    propagation across the await boundary."""

    responses: list[LLMResponse] = Field(default_factory=list)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _seen_run_ids: list[str] = PrivateAttr(default_factory=list)

    async def _complete(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        ctx = current_context()
        self._seen_run_ids.append(ctx.run_id if ctx is not None else "")
        self._calls.append({"messages": [dict(m) for m in messages]})
        if not self.responses:
            raise RuntimeError("MockLLMNode: ran out of scripted responses")
        return self.responses.pop(0)


def _resp(content: str = "ok", *, tool_calls: list[dict] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        model="m",
        role="assistant",
        usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


# ---------------------------------------------------------------------------
# RunContext / ContextVar basics
# ---------------------------------------------------------------------------


def test_current_context_returns_none_outside_run():
    assert current_context() is None


def test_push_sets_and_resets_context():
    assert current_context() is None
    ctx = RunContext(agent="alpha")
    with push(ctx):
        assert current_context() is ctx
    assert current_context() is None


def test_push_restores_previous_context_on_exception():
    outer = RunContext(agent="outer")
    inner = RunContext(agent="inner")
    with push(outer):
        try:
            with push(inner):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # Inner context popped even on raise.
        assert current_context() is outer
    assert current_context() is None


def test_derive_inherits_run_id_and_overrides_fields():
    parent = RunContext(agent="alpha", supervisor="planner", depth=1)
    with push(parent):
        child = derive(worker="researcher", depth=2)
        assert child.run_id == parent.run_id
        assert child.supervisor == "planner"  # inherited
        assert child.worker == "researcher"  # overridden
        assert child.depth == 2  # overridden


def test_derive_with_no_parent_mints_fresh_run_id():
    ctx = derive(agent="solo")
    assert ctx.run_id  # non-empty
    assert ctx.agent == "solo"
    assert ctx.supervisor is None


def test_context_propagates_across_await():
    """ContextVars propagate across `await` and `create_task` by
    default in asyncio. This guards against accidental regressions."""

    async def _child() -> str | None:
        ctx = current_context()
        return ctx.run_id if ctx is not None else None

    async def _parent() -> str | None:
        with push(RunContext(agent="parent")):
            return await _child()

    seen = asyncio.run(_parent())
    assert seen is not None and len(seen) > 0


def test_context_propagates_across_thread_pool_via_tools_node():
    """:class:`ToolsNode.fire` snapshots the caller's contextvars so
    tools running on worker threads see the active RunContext."""
    from llmagpie.base.node import MakeNode
    from llmagpie.base.tools import ToolsNode

    captured: dict[str, str | None] = {}

    @MakeNode.from_function(outputs={"out": str})
    def probe(marker: str) -> dict:
        """Capture the active RunContext from the executor thread."""
        ctx = current_context()
        captured["run_id"] = ctx.run_id if ctx is not None else None
        captured["marker"] = marker
        return {"out": "ok"}

    tools_node = ToolsNode(name="probe_node", tools=[probe])
    sentinel_ctx = RunContext(agent="caller")

    async def _run():
        with push(sentinel_ctx):
            await tools_node.async_call_(
                tool_calls_list=[
                    {"function": {"name": "probe", "arguments": '{"marker": "hi"}'}}
                ]
            )

    asyncio.run(_run())
    assert captured["run_id"] == sentinel_ctx.run_id


# ---------------------------------------------------------------------------
# Agent / Supervisor integration
# ---------------------------------------------------------------------------


def test_agent_run_populates_result_run_context():
    llm = MockLLMNode(name="m", responses=[_resp("hello")])
    agent = Agent(llm=llm, model="m", name="solo")

    result = asyncio.run(agent.run("hi"))
    assert result.run_context is not None
    assert result.run_context.agent == "solo"
    # The mock observed the active run_id at LLM time → must match.
    assert llm._seen_run_ids == [result.run_context.run_id]


def test_budget_exceeded_carries_run_context_on_solo_agent():
    llm = MockLLMNode(
        name="m",
        responses=[
            LLMResponse(
                content="",
                tool_calls=[],
                finish_reason="stop",
                model="m",
                role="assistant",
                usage=LLMUsage(prompt_tokens=99, completion_tokens=99, total_tokens=198),
            )
        ],
    )
    agent = Agent(llm=llm, model="m", name="solo", max_tokens_per_run=10)

    with pytest.raises(BudgetExceededError) as exc_info:
        asyncio.run(agent.run("hi"))
    assert exc_info.value.run_context is not None
    assert exc_info.value.run_context.agent == "solo"


def test_supervisor_attaches_delegation_trace_to_budget_error():
    """When the supervisor trips its budget mid-delegation, the
    raised :class:`BudgetExceededError` carries a ``run_context``
    whose ``delegation_trace`` reflects the calls made before the
    failure."""
    worker_llm = MockLLMNode(name="wm", responses=[_resp("done by worker")])
    inner = Agent(llm=worker_llm, model="m", name="researcher")

    sup_llm = MockLLMNode(
        name="sm",
        responses=[
            # First turn: supervisor delegates to the worker.
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "transfer_to_researcher",
                            "arguments": '{"task": "find facts"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
                model="m",
                role="assistant",
                usage=LLMUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100),
            ),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="m",
        workers=[inner.as_worker(name="researcher", description="Find facts.")],
        max_tokens_per_run=80,  # trips after the first round trip
        name="planner",
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        asyncio.run(sup.run("write a brief"))

    ctx = exc_info.value.run_context
    assert ctx is not None
    assert ctx.supervisor == "planner"
    assert ctx.delegation_trace is not None
    assert ctx.delegation_trace.worker == "planner"


def test_attach_context_is_idempotent():
    inner = RunContext(agent="inner")
    outer = RunContext(agent="outer")
    exc = RuntimeError("boom")
    attach_context(exc, inner)
    attach_context(exc, outer)  # should be a no-op since inner is set
    assert exc.run_context is inner  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_format_error_without_context_returns_header_only():
    out = format_error(RuntimeError("boom"))
    assert out == "RuntimeError: boom"


def test_format_error_with_context_renders_block():
    exc = RuntimeError("boom")
    exc.run_context = RunContext(  # type: ignore[attr-defined]
        run_id="7af3c1b8abcdef",
        agent="writer",
        supervisor="planner",
        worker="writer",
        depth=2,
        thread_id="thread_alpha",
    )
    out = format_error(exc)
    assert "RuntimeError: boom" in out
    assert "run_id" in out and "7af3c1b8" in out
    assert "agent" in out and "writer" in out
    assert "supervisor" in out and "planner" in out
    assert "worker" in out and "depth 2" in out
    assert "thread_alpha" in out


def test_format_error_renders_delegation_trace():
    from llmagpie.experimental.orchestration import DelegationTrace

    trace = DelegationTrace(
        worker="planner",
        task="brief on Mamba",
        depth=0,
        started_at=0.0,
        ended_at=1.2,
        usage=LLMUsage(total_tokens=15234),
    )
    trace.children.append(
        DelegationTrace(
            worker="writer",
            task="draft summary",
            depth=1,
            started_at=0.1,
            ended_at=0.9,
            usage=LLMUsage(total_tokens=8200),
        )
    )
    exc = RuntimeError("budget")
    exc.run_context = RunContext(  # type: ignore[attr-defined]
        run_id="abcdef01234567",
        supervisor="planner",
        delegation_trace=trace,
    )
    out = format_error(exc)
    assert "Delegation trace:" in out
    assert "planner" in out
    assert "writer" in out
    assert "draft summary" in out


def test_format_error_surfaces_budget_extras():
    """``BudgetExceededError`` carries ``budget_limit`` /
    ``budget_dimension`` / ``usage_so_far`` — ``format_error`` should
    surface those inline."""
    exc = BudgetExceededError(
        "solo: run exceeded max_tokens_per_run (200 > 10)",
        usage_so_far=LLMUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200),
        budget_limit=10.0,
        budget_dimension="tokens",
    )
    exc.run_context = RunContext(agent="solo")
    out = format_error(exc)
    assert "budget: 200 / 10.0 (tokens)" in out


def test_format_trace_returns_string_for_delegation_trace():
    from llmagpie.experimental.orchestration import DelegationTrace

    trace = DelegationTrace(
        worker="root", task="t", depth=0, started_at=0.0, ended_at=0.5
    )
    out = format_trace(trace)
    assert "root" in out
    assert out == trace.format()


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------


def test_logger_emits_run_id_and_agent_fields(caplog):
    """The default formatter references ``%(run_id)s`` etc.; verify
    the filter populates them (no KeyError) and that values match the
    active context."""
    logger = get_or_create_logger("test_observability_logger")
    with push(RunContext(run_id="deadbeef0123", agent="solo")):
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("hello from agent")
    # caplog captures the raw record; verify the filter ran by reading
    # the attributes back off the record.
    assert any(
        getattr(rec, "run_id", None) == "deadbeef" and getattr(rec, "agent", None) == "solo"
        for rec in caplog.records
    )


def test_logger_uses_placeholders_when_no_context(caplog):
    logger = get_or_create_logger("test_observability_logger_no_ctx")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("orphan log")
    assert any(
        getattr(rec, "run_id", None) == "-" and getattr(rec, "agent", None) == "-"
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


def test_json_formatter_emits_valid_json_with_context_fields():
    """Direct formatter test: emits a single-line JSON object with
    ts/level/logger/msg/run_id/agent/worker/depth populated."""
    import io
    import json as json_lib

    from llmagpie.observability import JsonFormatter, RunContextFilter

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test_json_formatter")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addFilter(RunContextFilter())

    with push(RunContext(run_id="abcdef0123456789", agent="solo", worker="alpha", depth=1)):
        logger.info("structured log line")

    line = buf.getvalue().strip().splitlines()[-1]
    payload = json_lib.loads(line)
    assert payload["msg"] == "structured log line"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test_json_formatter"
    assert payload["run_id"] == "abcdef01"
    assert payload["agent"] == "solo"
    assert payload["worker"] == "alpha"
    assert payload["depth"] == 1


def test_json_formatter_passes_through_extras():
    """Caller-supplied ``extra=`` fields get top-level keys in the
    JSON payload (so users can ship structured fields without
    subclassing)."""
    import io
    import json as json_lib

    from llmagpie.observability import JsonFormatter

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test_json_extras")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("with extras", extra={"latency_ms": 42, "user": "alice"})

    payload = json_lib.loads(buf.getvalue().strip().splitlines()[-1])
    assert payload["latency_ms"] == 42
    assert payload["user"] == "alice"


def test_resolve_formatter_picks_json_when_flag_set():
    """``json=True`` (explicit kwarg) selects the JsonFormatter."""
    from llmagpie.base.logging.logging import _resolve_formatter
    from llmagpie.observability import JsonFormatter

    assert isinstance(_resolve_formatter(True), JsonFormatter)
    assert not isinstance(_resolve_formatter(False), JsonFormatter)


def test_resolve_formatter_honors_env_var(monkeypatch):
    """``LLMAGPIE_LOG_JSON=1`` flips the formatter when no explicit arg."""
    from llmagpie.base.logging.logging import _resolve_formatter
    from llmagpie.observability import JsonFormatter

    monkeypatch.setenv("LLMAGPIE_LOG_JSON", "1")
    assert isinstance(_resolve_formatter(None), JsonFormatter)
    # Explicit False still wins over env var.
    assert not isinstance(_resolve_formatter(False), JsonFormatter)

    monkeypatch.setenv("LLMAGPIE_LOG_JSON", "0")
    assert not isinstance(_resolve_formatter(None), JsonFormatter)
