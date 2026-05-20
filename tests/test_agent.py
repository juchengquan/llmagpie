"""Tests for the high-level Agent abstraction.

The Agent composes BaseLLMNode + memory + cache + tools + (optional)
structured outputs. These tests use a stub BaseLLMNode that records
calls and returns canned LLMResponses; no real network."""

import asyncio
from typing import Any

import pytest
from llmagpie.base.node import MakeNode
from llmagpie.experimental.agent import Agent, AgentResult
from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from llmagpie.experimental.nodes.generators.cache import InMemoryCache
from llmagpie.experimental.nodes.generators.memory import InMemoryStore
from llmagpie.experimental.nodes.generators.structured import StructuredOutputError
from pydantic import BaseModel


class _RecorderLLM(BaseLLMNode):
    """A BaseLLMNode that records every (model, messages, kwargs)
    triple and yields canned responses from a sequence."""

    def _set_script(self, script: list[LLMResponse]) -> None:
        # Stash on the instance via Pydantic's __dict__ side-channel so
        # we don't have to declare it as a model field.
        object.__setattr__(self, "_script", list(script))
        object.__setattr__(self, "_calls", [])

    async def _complete(self, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        self._calls.append({"model": model, "messages": list(messages), "kwargs": dict(kwargs)})
        if not self._script:
            return LLMResponse(content="(default)")
        return self._script.pop(0)


def _resp(content: str = "ok", **extra: Any) -> LLMResponse:
    """Build an LLMResponse with default token usage so cumulative
    sums in AgentResult are non-trivial."""
    usage = extra.pop("usage", LLMUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8))
    return LLMResponse(content=content, usage=usage, **extra)


# ---------------------------------------------------------------------------
# Plain run: no memory, no cache, no tools, no schema.
# ---------------------------------------------------------------------------


def test_agent_run_returns_terminal_response_and_usage():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("hello world")])

    agent = Agent(llm=llm, model="m", system_prompt="be brief")

    result = asyncio.run(agent.run("hi"))

    assert isinstance(result, AgentResult)
    assert result.content == "hello world"
    assert result.usage.total_tokens == 8
    assert result.parsed is None
    # The system prompt is prepended to the user message.
    msgs = llm._calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert msgs[-1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# Memory: history persists across run() calls under one thread_id.
# ---------------------------------------------------------------------------


def test_agent_with_memory_accumulates_history_across_calls():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("hi alice"), _resp("you are alice")])

    agent = Agent(llm=llm, model="m", system_prompt="sys", memory_store=InMemoryStore())

    asyncio.run(agent.run("my name is alice", thread_id="t1"))
    asyncio.run(agent.run("who am i?", thread_id="t1"))

    # Second call's messages include the first turn's user + assistant
    # exchange, then the new user message.
    second = llm._calls[1]["messages"]
    contents = [m.get("content") for m in second]
    assert "my name is alice" in contents
    assert "hi alice" in contents
    assert "who am i?" in contents


def test_agent_threads_are_isolated():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("for alice"), _resp("for bob")])
    agent = Agent(llm=llm, model="m", memory_store=InMemoryStore())

    asyncio.run(agent.run("alice talking", thread_id="alice"))
    asyncio.run(agent.run("bob talking", thread_id="bob"))

    # Bob's second-call messages must NOT contain Alice's input.
    bob_messages = llm._calls[1]["messages"]
    assert all("alice" not in m.get("content", "") for m in bob_messages)


def test_agent_clear_history_drops_thread():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("r1"), _resp("r2")])
    agent = Agent(llm=llm, model="m", memory_store=InMemoryStore())

    asyncio.run(agent.run("first", thread_id="t1"))
    asyncio.run(agent.clear_history("t1"))
    asyncio.run(agent.run("after clear", thread_id="t1"))

    # The post-clear call should have no traces of "first".
    msgs = llm._calls[1]["messages"]
    assert all("first" not in m.get("content", "") for m in msgs)


# ---------------------------------------------------------------------------
# Cache: identical calls short-circuit the inner LLM.
# ---------------------------------------------------------------------------


def test_agent_with_cache_serves_repeat_calls_without_hitting_inner():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("first")])

    agent = Agent(llm=llm, model="m", cache=InMemoryCache())

    r1 = asyncio.run(agent.run("ping"))
    r2 = asyncio.run(agent.run("ping"))

    assert r1.content == r2.content == "first"
    # Inner LLM was only hit once.
    assert len(llm._calls) == 1


# ---------------------------------------------------------------------------
# Tools: the agent's bound tools are dispatched via the LLM tool-call loop.
# ---------------------------------------------------------------------------


def test_agent_runs_tool_call_loop_to_completion():
    llm = _RecorderLLM(name="recorder")
    # Round 1: ask to call echo. Round 2: return final answer.
    llm._set_script(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value": "hi"}'},
                    }
                ],
                finish_reason="tool_calls",
                usage=LLMUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            ),
            _resp("all done"),
        ]
    )

    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo the value."""
        return value

    agent = Agent(llm=llm, model="m", tools=[_echo])

    result = asyncio.run(agent.run("call echo"))

    assert result.content == "all done"
    assert result.tool_calls == []
    # Two LLM round-trips: the initial call + the post-tool follow-up.
    assert len(llm._calls) == 2
    # Cumulative usage from both rounds.
    assert result.usage.total_tokens == 6 + 8


# ---------------------------------------------------------------------------
# Structured outputs: response_schema parses + validates + self-repairs.
# ---------------------------------------------------------------------------


class _Weather(BaseModel):
    city: str
    temp_c: float


def test_agent_with_schema_returns_parsed_model():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp('{"city": "Paris", "temp_c": 12.0}')])

    agent = Agent(llm=llm, model="m", response_schema=_Weather)

    result = asyncio.run(agent.run("weather in Paris"))
    assert isinstance(result.parsed, _Weather)
    assert result.parsed.city == "Paris"
    assert result.parsed.temp_c == 12.0


def test_agent_with_schema_self_repairs_invalid_first_attempt():
    llm = _RecorderLLM(name="recorder")
    llm._set_script(
        [
            _resp("totally not json"),
            _resp('{"city": "Berlin", "temp_c": 7.5}'),
        ]
    )

    agent = Agent(llm=llm, model="m", response_schema=_Weather, repair_attempts=1)

    result = asyncio.run(agent.run("weather in Berlin"))
    assert result.parsed is not None
    assert result.parsed.city == "Berlin"
    # Two LLM round-trips total (initial + one repair).
    assert len(llm._calls) == 2
    # Cumulative usage adds both attempts.
    assert result.usage.total_tokens == 16


def test_agent_with_schema_raises_after_exhausted_repair():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("nope"), _resp("still nope")])

    agent = Agent(llm=llm, model="m", response_schema=_Weather, repair_attempts=1)

    with pytest.raises(StructuredOutputError):
        asyncio.run(agent.run("weather"))


# ---------------------------------------------------------------------------
# Custom params + caller-supplied message list bypassing the system prompt.
# ---------------------------------------------------------------------------


def test_agent_run_accepts_messages_list_and_keeps_existing_system_prompt():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("ok")])
    agent = Agent(llm=llm, model="m", system_prompt="X")

    asyncio.run(
        agent.run([{"role": "system", "content": "OVERRIDE"}, {"role": "user", "content": "hi"}])
    )

    # The caller's system message is preserved; the agent's default is
    # NOT inserted because the caller already supplied a system role.
    msgs = llm._calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "OVERRIDE"}
    assert {"role": "user", "content": "hi"} in msgs


def test_agent_run_forwards_params_to_provider():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("ok")])
    agent = Agent(llm=llm, model="m", memory_store=InMemoryStore())

    asyncio.run(agent.run("hi", params={"temperature": 0.1, "max_tokens": 200}))

    kwargs = llm._calls[0]["kwargs"]
    # `thread_id` is the only param consumed by MemoryNode; everything
    # else passes through to the inner LLM as kwargs.
    assert kwargs.get("temperature") == 0.1
    assert kwargs.get("max_tokens") == 200
    assert "thread_id" not in kwargs  # MemoryNode pops it before forwarding


# ---------------------------------------------------------------------------
# Misuse: provider that yields no responses must surface a clear error.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Budget enforcement: max_tokens_per_run / max_cost_per_run.
# ---------------------------------------------------------------------------


def test_agent_raises_budget_exceeded_on_token_ceiling():
    from llmagpie.experimental.agent import BudgetExceededError

    llm = _RecorderLLM(name="recorder")
    # Single response with 10 total_tokens; budget is 5 → trip.
    llm._set_script(
        [
            LLMResponse(
                content="big", usage=LLMUsage(prompt_tokens=4, completion_tokens=6, total_tokens=10)
            )
        ]
    )
    agent = Agent(llm=llm, model="m", max_tokens_per_run=5)

    with pytest.raises(BudgetExceededError) as ei:
        asyncio.run(agent.run("hi"))
    assert ei.value.budget_dimension == "tokens"
    assert ei.value.usage_so_far.total_tokens == 10


def test_agent_raises_budget_exceeded_across_tool_loop():
    """Budget is checked AFTER each round-trip; cumulative usage across
    the tool-call loop is what gets compared."""
    from llmagpie.experimental.agent import BudgetExceededError

    llm = _RecorderLLM(name="recorder")
    llm._set_script(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value": "x"}'},
                    }
                ],
                finish_reason="tool_calls",
                usage=LLMUsage(prompt_tokens=3, completion_tokens=3, total_tokens=6),
            ),
            LLMResponse(
                content="settled",
                usage=LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            ),
        ]
    )

    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo."""
        return value

    agent = Agent(llm=llm, model="m", tools=[_echo], max_tokens_per_run=10)

    with pytest.raises(BudgetExceededError):
        # First round: 6 tokens (OK). Second round: 6+10=16 > 10 → trips.
        asyncio.run(agent.run("call echo"))


def test_agent_max_cost_per_run_uses_price_table():
    from llmagpie.experimental.agent import BudgetExceededError

    llm = _RecorderLLM(name="recorder")
    llm._set_script(
        [
            LLMResponse(
                content="x",
                usage=LLMUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000),
            )
        ]
    )
    # $0.003 prompt + $0.015 completion per 1k → $0.018 total.
    agent = Agent(
        llm=llm,
        model="m",
        cost_per_1k_tokens={"prompt": 0.003, "completion": 0.015},
        max_cost_per_run=0.01,
    )
    with pytest.raises(BudgetExceededError) as ei:
        asyncio.run(agent.run("hi"))
    assert ei.value.budget_dimension == "cost"


def test_agent_cost_of_returns_zero_without_price_table():
    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("ok")])
    agent = Agent(llm=llm, model="m")
    assert agent.cost_of(LLMUsage(prompt_tokens=999, completion_tokens=999)) == 0.0


# ---------------------------------------------------------------------------
# Semantic stop conditions: factory helpers + driver-loop integration.
# ---------------------------------------------------------------------------


def test_stop_on_content_match_breaks_tool_loop():
    from llmagpie.experimental.nodes.generators.stop import stop_on_content_match

    llm = _RecorderLLM(name="recorder")
    # Round 1: tool call. Round 2: says DONE → stop here, no round 3.
    llm._set_script(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value": "x"}'},
                    }
                ],
                finish_reason="tool_calls",
            ),
            _resp("Result is X. DONE"),
            _resp("should-never-be-reached"),
        ]
    )

    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo."""
        return value

    agent = Agent(
        llm=llm,
        model="m",
        tools=[_echo],
        stop_condition=stop_on_content_match(r"\bDONE\b"),
    )
    result = asyncio.run(agent.run("go"))
    assert "DONE" in result.content
    assert len(llm._calls) == 2  # third scripted response wasn't consumed


def test_stop_on_tool_name_stops_on_specific_tool_call():
    from llmagpie.experimental.nodes.generators.stop import stop_on_tool_name

    llm = _RecorderLLM(name="recorder")
    llm._set_script(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "submit_answer", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
            ),
        ]
    )
    agent = Agent(
        llm=llm,
        model="m",
        stop_condition=stop_on_tool_name("submit_answer"),
    )
    result = asyncio.run(agent.run("go"))
    # The very first response triggered the stop — no tool dispatch, no
    # second LLM call. Single round-trip total.
    assert len(llm._calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "submit_answer"


def test_stop_on_finish_reason_aborts_truncated_runs():
    from llmagpie.experimental.nodes.generators.stop import stop_on_finish_reason

    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("got cut off", finish_reason="length"), _resp("never reached")])
    agent = Agent(llm=llm, model="m", stop_condition=stop_on_finish_reason("length"))
    result = asyncio.run(agent.run("go"))
    assert result.content == "got cut off"
    assert len(llm._calls) == 1


def test_any_of_combines_multiple_conditions():
    from llmagpie.experimental.nodes.generators.stop import (
        any_of,
        stop_on_content_match,
        stop_on_finish_reason,
    )

    llm = _RecorderLLM(name="recorder")
    llm._set_script([_resp("normal answer", finish_reason="length")])
    agent = Agent(
        llm=llm,
        model="m",
        stop_condition=any_of(
            stop_on_content_match(r"DONE"),
            stop_on_finish_reason("length"),
        ),
    )
    asyncio.run(agent.run("go"))
    assert len(llm._calls) == 1  # the length finish_reason matched


def test_agent_raises_when_llm_yields_nothing():
    class _BrokenLLM(BaseLLMNode):
        async def async_call(self, model, messages, **kwargs):
            # Async generator that immediately exits — no yields at all.
            if False:
                yield  # pragma: no cover

    llm = _BrokenLLM(name="broken")
    agent = Agent(llm=llm, model="m")

    with pytest.raises(RuntimeError, match="zero responses"):
        asyncio.run(agent.run("hi"))
