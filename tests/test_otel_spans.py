"""Phase-2 tests: GenAI semconv spans.

Uses :class:`InMemorySpanExporter` (no OTel collector required) to
register a tracer provider, run an :class:`Agent` /
:class:`Supervisor`, and assert that the span tree has the right
shape and the GenAI attributes are present.

The fixture installs the in-memory provider once per session and
swaps the exporter between tests so spans don't leak across tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from llmagpie.experimental.agent import Agent
from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from llmagpie.experimental.orchestration import Supervisor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field, PrivateAttr

# ---------------------------------------------------------------------------
# Provider setup — InMemoryExporter so we can introspect emitted spans.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def otel_exporter() -> InMemorySpanExporter:
    """Install an in-memory tracer provider for the session and return
    the exporter so tests can drain it.

    OTel allows ``set_tracer_provider`` to be called only once per
    process; the session scope respects that. Tests clear the
    exporter at the start of each test via the ``clear_spans`` fixture.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Some other test may have installed a provider; OTel logs a warning
    # but accepts the override.
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture(autouse=True)
def _clear_spans(otel_exporter: InMemorySpanExporter):
    otel_exporter.clear()
    yield
    otel_exporter.clear()


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLMNode(BaseLLMNode):
    """LLM that replays a pre-scripted list of LLMResponses."""

    responses: list[LLMResponse] = Field(default_factory=list)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    async def _complete(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        self._calls.append({"messages": [dict(m) for m in messages]})
        if not self.responses:
            raise RuntimeError("MockLLMNode: ran out of scripted responses")
        return self.responses.pop(0)


def _resp(content: str = "ok", *, tool_calls: list[dict] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        model="gpt-4",
        role="assistant",
        usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for span in exporter.get_finished_spans():
        out.setdefault(span.name, []).append(span)
    return out


# ---------------------------------------------------------------------------
# Agent spans
# ---------------------------------------------------------------------------


def test_agent_run_emits_invoke_agent_and_chat_spans(otel_exporter):
    llm = MockLLMNode(name="m", responses=[_resp("hello")])
    agent = Agent(llm=llm, model="gpt-4", name="solo")

    asyncio.run(agent.run("hi"))

    spans = otel_exporter.get_finished_spans()
    names = [s.name for s in spans]
    # invoke_agent wraps the chat span — both should be present.
    assert any("invoke_agent solo" in n for n in names), names
    assert any("chat gpt-4" in n for n in names), names


def test_chat_span_has_gen_ai_attributes(otel_exporter):
    llm = MockLLMNode(name="m", responses=[_resp("hello")])
    agent = Agent(llm=llm, model="gpt-4", name="solo")

    asyncio.run(agent.run("hi"))

    chat = next(s for s in otel_exporter.get_finished_spans() if s.name.startswith("chat"))
    attrs = dict(chat.attributes or {})
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.request.model") == "gpt-4"
    assert attrs.get("gen_ai.response.model") == "gpt-4"
    assert attrs.get("gen_ai.usage.input_tokens") == 10
    assert attrs.get("gen_ai.usage.output_tokens") == 20
    # finish_reasons is a list-typed attribute in semconv.
    assert list(attrs.get("gen_ai.response.finish_reasons") or []) == ["stop"]
    # MockLLMNode → "mockllm" after the suffix-strip rule.
    assert attrs.get("gen_ai.system") == "mockllm"
    # session.id correlates the span to the RunContext.
    assert isinstance(attrs.get("session.id"), str)


def test_agent_span_has_openinference_kind_and_session(otel_exporter):
    llm = MockLLMNode(name="m", responses=[_resp("hello")])
    agent = Agent(llm=llm, model="gpt-4", name="solo")

    asyncio.run(agent.run("hi"))

    agent_span_obj = next(
        s for s in otel_exporter.get_finished_spans() if s.name.startswith("invoke_agent")
    )
    attrs = dict(agent_span_obj.attributes or {})
    assert attrs.get("openinference.span.kind") == "AGENT"
    assert attrs.get("gen_ai.agent.name") == "solo"
    assert attrs.get("gen_ai.operation.name") == "invoke_agent"
    assert isinstance(attrs.get("session.id"), str)


# ---------------------------------------------------------------------------
# Supervisor / handoff spans
# ---------------------------------------------------------------------------


def test_supervisor_emits_handoff_and_nested_agent_spans(otel_exporter):
    """A supervisor delegating to a single worker should emit:
    supervise / handoff / invoke_agent / chat spans, with the
    handoff parented to the supervise span and the worker's
    invoke_agent parented to the handoff."""
    worker_llm = MockLLMNode(name="wm", responses=[_resp("done")])
    inner = Agent(llm=worker_llm, model="gpt-4", name="researcher")

    sup_llm = MockLLMNode(
        name="sm",
        responses=[
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
                model="gpt-4",
                role="assistant",
                usage=LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            ),
            _resp("brief written"),
        ],
    )

    sup = Supervisor(
        llm=sup_llm,
        model="gpt-4",
        workers=[inner.as_worker(name="researcher", description="Find facts.")],
        name="planner",
    )

    asyncio.run(sup.run("write a brief"))

    spans = otel_exporter.get_finished_spans()
    by_name = _spans_by_name(otel_exporter)

    assert any(n.startswith("supervise planner") for n in by_name), list(by_name)
    assert any(n.startswith("handoff planner→researcher") for n in by_name), list(by_name)
    assert any(n.startswith("invoke_agent researcher") for n in by_name), list(by_name)

    handoff = next(s for s in spans if s.name.startswith("handoff"))
    attrs = dict(handoff.attributes or {})
    assert attrs.get("openinference.span.kind") == "CHAIN"
    assert attrs.get("gen_ai.operation.name") == "handoff"
    assert attrs.get("llmagpie.handoff.source") == "planner"
    assert attrs.get("llmagpie.handoff.target") == "researcher"
    assert attrs.get("llmagpie.handoff.depth") == 1
    # task_preview truncates at 120 chars; "find facts" fits whole.
    assert attrs.get("llmagpie.handoff.task_preview") == "find facts"


def test_handoff_span_parents_worker_agent_span(otel_exporter):
    """The handoff span should be the parent of the worker's
    invoke_agent span (and the supervise span should parent the
    handoff)."""
    worker_llm = MockLLMNode(name="wm", responses=[_resp("done")])
    inner = Agent(llm=worker_llm, model="gpt-4", name="researcher")
    sup_llm = MockLLMNode(
        name="sm",
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "transfer_to_researcher",
                            "arguments": '{"task": "go"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
                model="gpt-4",
                role="assistant",
                usage=LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            ),
            _resp("done"),
        ],
    )
    sup = Supervisor(
        llm=sup_llm,
        model="gpt-4",
        workers=[inner.as_worker(name="researcher", description="...")],
        name="planner",
    )
    asyncio.run(sup.run("hi"))

    spans = otel_exporter.get_finished_spans()
    handoff = next(s for s in spans if s.name.startswith("handoff"))
    worker_invoke = next(
        s for s in spans if s.name.startswith("invoke_agent researcher")
    )
    supervise = next(s for s in spans if s.name.startswith("supervise"))

    # OpenTelemetry uses parent_span_id (uint64 form) — compare against
    # the parent's context.span_id.
    assert worker_invoke.parent.span_id == handoff.context.span_id
    assert handoff.parent.span_id == supervise.context.span_id


# ---------------------------------------------------------------------------
# Tool spans
# ---------------------------------------------------------------------------


def test_tool_invocation_emits_tool_span_parented_under_agent(otel_exporter):
    """A tool dispatched via ToolsNode runs on a worker thread; the
    span helper opens the tool span *inside* the thread so its parent
    is the agent/chat span via the copied OTel context."""
    from llmagpie.base.node import MakeNode

    @MakeNode.from_function(outputs={"out": str})
    def upper(value: str) -> dict:
        """Uppercase the input."""
        return {"out": value.upper()}

    llm = MockLLMNode(
        name="m",
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "upper", "arguments": '{"value": "abc"}'},
                    }
                ],
                finish_reason="tool_calls",
                model="gpt-4",
                role="assistant",
                usage=LLMUsage(prompt_tokens=3, completion_tokens=3, total_tokens=6),
            ),
            _resp("done"),
        ],
    )
    agent = Agent(llm=llm, model="gpt-4", tools=[upper], name="solo")
    asyncio.run(agent.run("hi"))

    spans = otel_exporter.get_finished_spans()
    tool = next((s for s in spans if s.name.startswith("execute_tool")), None)
    assert tool is not None, [s.name for s in spans]
    attrs = dict(tool.attributes or {})
    assert attrs.get("openinference.span.kind") == "TOOL"
    assert attrs.get("gen_ai.tool.name") == "upper"

    # The tool span's parent should be the agent_span (since the tool
    # is dispatched outside any chat span — chat opens/closes per
    # _complete call).
    invoke = next(s for s in spans if s.name.startswith("invoke_agent"))
    assert tool.parent.span_id == invoke.context.span_id


# ---------------------------------------------------------------------------
# No-op fallback
# ---------------------------------------------------------------------------


def test_no_op_when_otel_module_missing(monkeypatch):
    """When the ``opentelemetry`` import is unavailable the helpers
    must return a no-op span that accepts every method call without
    raising. We simulate the missing-OTel branch by overriding
    ``_tracer`` to return None."""
    from llmagpie.observability import _otel

    monkeypatch.setattr(_otel, "_tracer", lambda: None)

    with _otel.chat_span(model="gpt-4", system="openai") as span:
        # Every operation must be safe on the null span.
        span.set_attribute("k", "v")
        span.set_attributes({"k2": "v2"})
        _otel.set_llm_attributes(span, model="gpt-4")
        span.record_exception(RuntimeError("x"))

    with _otel.agent_span(agent_name="a") as span:
        span.set_attribute("k", "v")

    with _otel.handoff_span(source="s", target="t", task="x", depth=1) as span:
        span.set_attribute("k", "v")

    with _otel.tool_span(tool_name="t") as span:
        span.set_attribute("k", "v")
