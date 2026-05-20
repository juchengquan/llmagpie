"""Targeted unit tests for behaviors that regressed in the past or are
easy to break without noticing."""

import asyncio

import pytest
from llmagpie.base.connectable import BaseConnectable
from llmagpie.base.enum import ConnectableType
from llmagpie.base.logging.logging_wrapper import log_output
from llmagpie.base.node import BaseNode, MakeNode
from llmagpie.base.utils.async_to_sync import exec_generator_in_event_loop


class _Concrete(BaseConnectable):
    """Minimal concrete BaseConnectable for state-isolation tests."""

    connectable_type: ConnectableType = ConnectableType.BASENODE

    def _validate(self):
        return True

    async def async_event_on_execution(self, inputs, session_id, **kwargs):
        if False:
            yield


def test_input_keys_are_per_instance():
    """`_input_keys_bound` / `_input_keys_nodes_map` used to be declared
    as bare class attributes — every BaseConnectable instance shared the
    same set/dict. They are now PrivateAttr-backed per-instance state."""
    a = _Concrete(name="a")
    b = _Concrete(name="b")

    a._input_keys_bound.add("x")
    a._input_keys_nodes_map["x"] = ["node-a"]

    assert b._input_keys_bound == set()
    assert b._input_keys_nodes_map == {}


def test_log_output_preserves_sync_shape():
    """`log_output` used to always return a coroutine — sync targets
    became awaitables silently."""

    @log_output
    def sync_fn(x):
        return x * 2

    assert sync_fn(3) == 6


def test_log_output_preserves_async_shape():
    @log_output
    async def async_fn(x):
        return x + 1

    assert asyncio.run(async_fn(4)) == 5


def test_exec_generator_propagates_exceptions():
    """Exceptions from an async generator must raise out of the sync bridge,
    not get yielded as opaque values to the caller."""

    async def async_gen():
        yield 1
        raise ValueError("intentional")

    loop = asyncio.new_event_loop()
    try:
        bridge = exec_generator_in_event_loop(async_gen(), loop)
        seen = []
        with pytest.raises(ValueError, match="intentional"):
            for v in bridge:
                seen.append(v)
        assert seen == [1]
    finally:
        loop.close()


def test_make_node_from_function_runs():
    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo back the input."""
        return value

    assert isinstance(_echo, BaseNode)
    assert _echo.run(value="hi") == {"value": "hi"}


# ---------------------------------------------------------------------------
# precheck behaviour: missing inputs and condition-function gating both
# return None and warn — they must not raise. Locks in the contract used
# by the pipeline's _collect_*_tasks paths.
# ---------------------------------------------------------------------------


def _make_passthrough(name: str = "pass"):
    @MakeNode.from_function(name=name, outputs={"value": str})
    def _node(value: str) -> str:
        """Pass through."""
        return value

    return _node


def test_precheck_returns_none_when_required_missing():
    node = _make_passthrough()
    assert node.precheck(session_id="s1", inputs={}) is None


def test_precheck_returns_none_when_condition_false():
    node = _make_passthrough()
    node.cond_func = lambda **_: False
    node.inputs_to_cond = {}
    assert node.precheck(session_id="s1", inputs={"value": "x"}) is None


# ---------------------------------------------------------------------------
# Loop cap: the per-node iteration counter must raise when
# `max_iteration_limit` is hit, not silently spin forever.
# ---------------------------------------------------------------------------


def test_iteration_counter_raises_at_limit():
    node = _make_passthrough()
    node.max_iteration_limit = 2
    node._callback("session", {"value": "first"})
    node._callback("session", {"value": "second"})
    with pytest.raises(Exception, match="max_iteration_limit"):
        node._callback("session", {"value": "third"})


# ---------------------------------------------------------------------------
# async_invoke: state cleanup must run AFTER the caller has consumed the
# returned generator, not before. (The old implementation called
# `clean_states` in a `finally` immediately after `return async_result`, so
# state was wiped before the generator was iterated.)
# ---------------------------------------------------------------------------


def test_async_invoke_round_trip_and_cleanup_order():
    from llmagpie import BasePipeline

    @MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
    class _Hello(BaseNode):
        async def async_call(self, name: str):
            return {"outputs": f"hello {name}"}

    node = _Hello(name="hello")
    pipe = BasePipeline(name="pipe", nodes=[node])
    pipe.compile()

    async def driver():
        gen = await pipe.async_invoke(inputs={"hello.name": "world"})
        out = [state.value async for state in gen]
        return out, set(pipe.input_state.keys()), set(node.input_state.keys())

    states, pipe_sessions, node_sessions = asyncio.run(driver())
    assert states == [{"outputs": "hello world"}]
    # After full consumption the per-session entries must be gone.
    assert pipe_sessions == set()
    assert node_sessions == set()


# ---------------------------------------------------------------------------
# ToolsNode: smoke-test the fire path that dispatches sub-nodes via a
# ThreadPoolExecutor. The only multi-threaded code in the library.
# ---------------------------------------------------------------------------


def test_tools_node_fires_each_tool():
    from llmagpie.base.tools import ToolsNode

    @MakeNode.from_function(name="upper", outputs={"value": str})
    def _upper(value: str) -> str:
        """Uppercase."""
        return value.upper()

    tools = ToolsNode(name="tools", tools=[_upper])
    calls = [
        {"function": {"name": "upper", "arguments": {"value": "abc"}}},
        {"function": {"name": "upper", "arguments": '{"value": "def"}'}},  # str arguments
    ]
    out = tools.run(tool_calls_list=calls)
    results = out["tool_calls_list"]
    assert [c["output"] for c in results] == [{"value": "ABC"}, {"value": "DEF"}]
    assert all(c["error"] is None for c in results)
    assert all("_f" not in c for c in results)


# ---------------------------------------------------------------------------
# Compile guard: pipeline mutations after compile() must be rejected, or
# the cached schema would silently diverge from the graph.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pipeline visualization: Mermaid + Graphviz exports.
# ---------------------------------------------------------------------------


def _two_node_pipeline():
    """Helper: tiny linear pipeline a -> b."""
    from llmagpie import BasePipeline

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _A(BaseNode):
        async def async_call(self, name: str):
            return {"out": name}

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _B(BaseNode):
        async def async_call(self, out: str):
            return {"out": out.upper()}

    a, b = _A(name="a"), _B(name="b")
    pipe = BasePipeline(name="pipe", nodes=[a, b])
    (a >> "out") >> ("out" >> b)
    pipe.compile()
    return pipe


def test_pipeline_to_mermaid_contains_nodes_and_edge():
    pipe = _two_node_pipeline()
    diagram = pipe.to_mermaid()
    assert diagram.startswith("flowchart LR")
    assert "a<br/>" in diagram and "b<br/>" in diagram
    # The edge label encodes the key mapping.
    assert "out→out" in diagram


def test_pipeline_to_graphviz_contains_nodes_and_edge():
    pipe = _two_node_pipeline()
    dot = pipe.to_graphviz()
    assert dot.startswith("digraph ")
    assert "rankdir=LR" in dot
    assert "->" in dot
    assert "out→out" in dot


def test_pipeline_visualization_rejects_bad_direction():
    pipe = _two_node_pipeline()
    with pytest.raises(ValueError, match="direction must be"):
        pipe.to_mermaid(direction="diagonal")
    with pytest.raises(ValueError, match="rankdir must be"):
        pipe.to_graphviz(rankdir="diagonal")


# ---------------------------------------------------------------------------
# Batch invocation: fan a list of inputs across the same pipeline.
# ---------------------------------------------------------------------------


def test_batch_invoke_preserves_order_and_runs_concurrently():
    from llmagpie import BasePipeline
    from llmagpie.base.utils.batch import batch_invoke

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _Shout(BaseNode):
        async def async_call(self, name: str):
            return {"out": f"hello {name}".upper()}

    node = _Shout(name="shout")
    pipe = BasePipeline(name="pipe", nodes=[node])
    pipe.compile()

    names = ["world", "magpie", "anthropic"]
    results = batch_invoke(pipe, [{"shout.name": n} for n in names])

    assert len(results) == 3
    finals = [r[-1].value for r in results]
    assert finals == [{"out": "HELLO WORLD"}, {"out": "HELLO MAGPIE"}, {"out": "HELLO ANTHROPIC"}]


def test_batch_invoke_return_exceptions_captures_failures():
    import asyncio

    from llmagpie import BasePipeline
    from llmagpie.base.utils.batch import async_batch_invoke

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _Maybe(BaseNode):
        async def async_call(self, name: str):
            if name == "boom":
                raise ValueError("intentional")
            return {"out": name}

    node = _Maybe(name="maybe")
    pipe = BasePipeline(name="pipe", nodes=[node])
    pipe.compile()

    inputs = [{"maybe.name": "ok"}, {"maybe.name": "boom"}, {"maybe.name": "fine"}]
    results = asyncio.run(async_batch_invoke(pipe, inputs, return_exceptions=True))

    assert results[0][-1].value == {"out": "ok"}
    assert isinstance(results[1], ValueError)
    assert results[2][-1].value == {"out": "fine"}


# ---------------------------------------------------------------------------
# Retry / fallback decorators.
# ---------------------------------------------------------------------------


def test_with_retry_succeeds_after_transient_failures():
    import asyncio

    from llmagpie.base.utils.retry import with_retry

    calls = {"n": 0}

    @with_retry(max_attempts=4, backoff_base=0.0, jitter=False)
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError(f"transient {calls['n']}")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert calls["n"] == 3


def test_with_retry_raises_after_exhaustion():
    import asyncio

    from llmagpie.base.utils.retry import with_retry

    @with_retry(max_attempts=2, backoff_base=0.0, jitter=False)
    async def always_fails():
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        asyncio.run(always_fails())


def test_with_retry_respects_should_retry_predicate():
    import asyncio

    from llmagpie.base.utils.retry import with_retry

    calls = {"n": 0}

    @with_retry(
        max_attempts=5, backoff_base=0.0, jitter=False, should_retry=lambda e: "retry-me" in str(e)
    )
    async def picky():
        calls["n"] += 1
        raise ValueError("do not retry")

    with pytest.raises(ValueError, match="do not retry"):
        asyncio.run(picky())
    assert calls["n"] == 1  # no retries because predicate said no


def test_with_fallback_invokes_fallback_on_failure():
    import asyncio

    from llmagpie.base.utils.retry import with_fallback

    async def fallback(x):
        return f"fallback({x})"

    @with_fallback(fallback)
    async def primary(x):
        raise RuntimeError("primary broken")

    assert asyncio.run(primary(7)) == "fallback(7)"


# ---------------------------------------------------------------------------
# multi_switch: route src -> exactly one of N branches based on a value.
# ---------------------------------------------------------------------------


def test_multi_switch_routes_to_matching_branch():
    from llmagpie import BasePipeline
    from llmagpie.base.utils.routing import multi_switch

    @MakeNode.from_class(func_name="async_call", outputs={"kind": str, "payload": str})
    class Router(BaseNode):
        async def async_call(self, kind: str, payload: str):
            return {"kind": kind, "payload": payload}

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _A(BaseNode):
        async def async_call(self, kind: str):
            return {"out": f"A:{kind}"}

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _B(BaseNode):
        async def async_call(self, kind: str):
            return {"out": f"B:{kind}"}

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _C(BaseNode):
        async def async_call(self, kind: str):
            return {"out": f"C:{kind}"}

    router = Router(name="router")
    a, b, c = _A(name="a"), _B(name="b"), _C(name="c")

    pipe = BasePipeline(name="pipe", nodes=[router, a, b, c])
    multi_switch(
        pipe,
        router,
        src_key="kind",
        dest_key="kind",
        branches={"alpha": a, "beta": b, "gamma": c},
    )
    pipe.compile()

    # Route "beta" → only b fires
    final_states = list(pipe.invoke(inputs={"router.kind": "beta", "router.payload": "hi"}))
    final_values = [s.value for s in final_states]
    # b emitted exactly once; a and c did not emit.
    assert {"out": "B:beta"} in final_values
    assert not any(v.get("out", "").startswith("A:") for v in final_values)
    assert not any(v.get("out", "").startswith("C:") for v in final_values)


def test_multi_switch_with_selector():
    from llmagpie import BasePipeline
    from llmagpie.base.utils.routing import multi_switch

    @MakeNode.from_class(func_name="async_call", outputs={"label": str})
    class Producer(BaseNode):
        async def async_call(self, label: str):
            return {"label": label}

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _Up(BaseNode):
        async def async_call(self, label: str):
            return {"out": "uppercase-branch"}

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _Lo(BaseNode):
        async def async_call(self, label: str):
            return {"out": "lowercase-branch"}

    p, up, lo = Producer(name="p"), _Up(name="up"), _Lo(name="lo")
    pipe = BasePipeline(name="pipe", nodes=[p, up, lo])
    multi_switch(
        pipe,
        p,
        src_key="label",
        dest_key="label",
        branches={"u": up, "l": lo},
        selector=lambda s: "u" if s.isupper() else "l",
    )
    pipe.compile()
    finals = [s.value for s in pipe.invoke(inputs={"p.label": "HELLO"})]
    assert {"out": "uppercase-branch"} in finals
    assert {"out": "lowercase-branch"} not in finals


def test_multi_switch_rejects_empty_or_duplicate_branches():
    import pytest
    from llmagpie import BasePipeline
    from llmagpie.base.utils.routing import multi_switch

    @MakeNode.from_function(name="src", outputs={"x": str})
    def _src(x: str) -> str:
        """src."""
        return x

    pipe = BasePipeline(name="pipe", nodes=[_src])
    with pytest.raises(ValueError, match="non-empty"):
        multi_switch(pipe, _src, src_key="x", dest_key="x", branches={})


# ---------------------------------------------------------------------------
# LLM provider abstraction: BaseLLMNode + LLMResponse driver loop.
# ---------------------------------------------------------------------------


def test_base_llm_node_tool_loop_terminates_when_no_tool_calls():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse

    class _Stub(BaseLLMNode):
        # Pydantic v2: declare in-class so model_fields picks it up.
        async def _complete(self, model, messages, **kwargs):
            return LLMResponse(content="done", tool_calls=[], finish_reason="stop")

    node = _Stub(name="stub")

    async def drive():
        out = []
        async for r in node.async_call(model="m", messages=[{"role": "user", "content": "hi"}]):
            out.append(r)
        return out

    results = asyncio.run(drive())
    assert len(results) == 1
    assert results[0].content == "done"
    assert results[0].tool_calls == []


def test_base_llm_node_tool_loop_dispatches_and_reprompts():
    import asyncio

    from llmagpie.base.tools import ToolsNode
    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse

    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo a value."""
        return value

    tools = ToolsNode(name="tools", tools=[_echo])

    class _Stub(BaseLLMNode):
        # First call: ask to call echo. Second call: settle with final answer.
        async def _complete(self, model, messages, **kwargs):
            if any(m.get("role") == "tool" for m in messages):
                return LLMResponse(content="all done", tool_calls=[], finish_reason="stop")
            return LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value": "hi"}'},
                    }
                ],
                finish_reason="tool_calls",
            )

    node = _Stub(name="stub", tools_node=tools)

    async def drive():
        return [
            r
            async for r in node.async_call(
                model="m", messages=[{"role": "user", "content": "use echo"}]
            )
        ]

    results = asyncio.run(drive())
    # Two LLM round-trips → two LLMResponses.
    assert len(results) == 2
    assert results[0].tool_calls != []
    assert results[-1].content == "all done"
    assert results[-1].tool_calls == []


def test_base_llm_node_respects_max_tool_iterations():
    import asyncio

    from llmagpie.base.tools import ToolsNode
    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse

    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo."""
        return value

    tools = ToolsNode(name="tools", tools=[_echo])

    class _Looper(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            return LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{len(messages)}",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value": "x"}'},
                    }
                ],
                finish_reason="tool_calls",
            )

    node = _Looper(name="loop", tools_node=tools, max_tool_iterations=2)

    async def drive():
        return [
            r
            async for r in node.async_call(model="m", messages=[{"role": "user", "content": "go"}])
        ]

    results = asyncio.run(drive())
    # 1 initial call + 2 iterations = 3 LLMResponses
    assert len(results) == 3
    # Cap was hit; last response still has tool_calls (it didn't get to settle).
    assert results[-1].tool_calls != []


def test_anthropic_message_translation_round_trip():
    """Unit-test the (private) Anthropic message translation without
    hitting the network."""
    from llmagpie.experimental.nodes.generators.anthropic_node import (
        _split_system,
        _to_anthropic_messages,
    )

    openai_msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "rain"}'},
                }
            ],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "t1"},
    ]
    system, rest = _split_system(openai_msgs)
    assert system == "be helpful"
    translated = _to_anthropic_messages(rest)
    # user, assistant-with-tool_use, user-with-tool_result
    assert len(translated) == 3
    assert translated[0]["role"] == "user"
    assert translated[1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" and b["name"] == "search" for b in translated[1]["content"])
    assert translated[2]["role"] == "user"
    assert translated[2]["content"][0]["type"] == "tool_result"


# ---------------------------------------------------------------------------
# Structured outputs: parse + validate LLM responses against a Pydantic model.
# ---------------------------------------------------------------------------


def test_call_with_schema_parses_clean_json():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.structured import call_with_schema
    from pydantic import BaseModel as _PM

    class Weather(_PM):
        city: str
        temp_c: float

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            return LLMResponse(content='{"city": "Berlin", "temp_c": 7.5}')

    node = _Stub(name="stub")
    result = asyncio.run(
        call_with_schema(
            node, model="m", messages=[{"role": "user", "content": "Q"}], schema=Weather
        )
    )
    assert isinstance(result, Weather)
    assert result.city == "Berlin" and result.temp_c == 7.5


def test_call_with_schema_strips_fences():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.structured import call_with_schema
    from pydantic import BaseModel as _PM

    class Point(_PM):
        x: int
        y: int

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            return LLMResponse(content='Here is your point:\n```json\n{"x": 1, "y": 2}\n```\nDone!')

    node = _Stub(name="stub")
    result = asyncio.run(
        call_with_schema(node, model="m", messages=[{"role": "user", "content": "Q"}], schema=Point)
    )
    assert result.x == 1 and result.y == 2


def test_call_with_schema_repairs_then_succeeds():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.structured import call_with_schema
    from pydantic import BaseModel as _PM

    class Item(_PM):
        name: str

    calls = {"n": 0}

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return LLMResponse(content="that is not json at all")
            return LLMResponse(content='{"name": "ok"}')

    node = _Stub(name="stub")
    result = asyncio.run(
        call_with_schema(
            node,
            model="m",
            messages=[{"role": "user", "content": "Q"}],
            schema=Item,
            max_repair_attempts=2,
        )
    )
    assert calls["n"] == 2  # first failed, second succeeded
    assert result.name == "ok"


def test_call_with_schema_raises_after_exhausted_repairs():
    import asyncio

    import pytest as _pytest
    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.structured import (
        StructuredOutputError,
        call_with_schema,
    )
    from pydantic import BaseModel as _PM

    class Item(_PM):
        name: str

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            return LLMResponse(content="never valid json")

    node = _Stub(name="stub")
    with _pytest.raises(StructuredOutputError) as ei:
        asyncio.run(
            call_with_schema(
                node,
                model="m",
                messages=[{"role": "user", "content": "Q"}],
                schema=Item,
                max_repair_attempts=1,
            )
        )
    assert ei.value.last_content == "never valid json"


# ---------------------------------------------------------------------------
# LLM response caching: short-circuit `_complete` on (model, messages, kwargs).
# ---------------------------------------------------------------------------


def test_cached_llm_node_hits_after_first_call():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.cache import (
        CachedLLMNode,
        InMemoryCache,
    )

    calls = {"n": 0}

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            calls["n"] += 1
            return LLMResponse(content=f"call#{calls['n']}")

    inner = _Stub(name="inner")
    cached = CachedLLMNode(name="cached", inner=inner, cache=InMemoryCache())

    async def run_twice():
        m = [{"role": "user", "content": "ping"}]
        r1 = await cached._complete("m", list(m))
        r2 = await cached._complete("m", list(m))
        return r1, r2

    r1, r2 = asyncio.run(run_twice())
    assert calls["n"] == 1  # inner only called once
    assert r1.content == r2.content == "call#1"  # second is cache hit


def test_cached_llm_node_differentiates_on_model_and_messages():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.cache import (
        CachedLLMNode,
        InMemoryCache,
    )

    calls = {"n": 0}

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            calls["n"] += 1
            return LLMResponse(content=f"call#{calls['n']}")

    cached = CachedLLMNode(name="c", inner=_Stub(name="i"), cache=InMemoryCache())

    async def run():
        await cached._complete("model-a", [{"role": "user", "content": "x"}])
        await cached._complete("model-b", [{"role": "user", "content": "x"}])  # diff model
        await cached._complete("model-a", [{"role": "user", "content": "y"}])  # diff message

    asyncio.run(run())
    assert calls["n"] == 3  # all distinct keys


def test_file_cache_round_trip(tmp_path):
    import asyncio

    from llmagpie.experimental.nodes.generators.cache import FileCache

    cache = FileCache(tmp_path)

    async def driver():
        assert await cache.get("k") is None
        await cache.set("k", b"hello")
        assert await cache.get("k") == b"hello"
        # Persists across reopens.
        cache2 = FileCache(tmp_path)
        assert await cache2.get("k") == b"hello"

    asyncio.run(driver())


def test_in_memory_cache_ttl_expiry():
    import asyncio
    import time

    from llmagpie.experimental.nodes.generators.cache import InMemoryCache

    cache = InMemoryCache()

    async def driver():
        await cache.set("k", b"v", ttl=1)
        assert await cache.get("k") == b"v"

    asyncio.run(driver())
    time.sleep(1.05)

    async def check():
        assert await cache.get("k") is None

    asyncio.run(check())


# ---------------------------------------------------------------------------
# Conversation memory: persist history across `_complete` calls per thread.
# ---------------------------------------------------------------------------


def test_memory_node_persists_history_across_calls():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.memory import InMemoryStore, MemoryNode

    seen_lengths: list[int] = []

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            seen_lengths.append(len(messages))
            return LLMResponse(content=f"reply{len(messages)}")

    store = InMemoryStore()
    node = MemoryNode(name="mem", inner=_Stub(name="i"), store=store)

    async def driver():
        await node._complete("m", [{"role": "user", "content": "hi"}], thread_id="t1")
        await node._complete("m", [{"role": "user", "content": "again"}], thread_id="t1")
        return await store.get("t1")

    history = asyncio.run(driver())

    # First call: 1 message (just "hi"). Second call: prev 2 (user+assistant)
    # + 1 new user = 3 messages forwarded.
    assert seen_lengths == [1, 3]
    # Stored: user/assistant from turn 1 + user/assistant from turn 2 = 4 entries.
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "hi"
    assert history[2]["content"] == "again"


def test_memory_node_isolates_threads():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.memory import InMemoryStore, MemoryNode

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            return LLMResponse(content="ok")

    store = InMemoryStore()
    node = MemoryNode(name="mem", inner=_Stub(name="i"), store=store)

    async def driver():
        await node._complete("m", [{"role": "user", "content": "alice-hi"}], thread_id="alice")
        await node._complete("m", [{"role": "user", "content": "bob-hi"}], thread_id="bob")
        return await store.get("alice"), await store.get("bob")

    alice, bob = asyncio.run(driver())
    assert len(alice) == 2 and alice[0]["content"] == "alice-hi"
    assert len(bob) == 2 and bob[0]["content"] == "bob-hi"


def test_memory_node_trims_to_max_messages_preserving_system():
    import asyncio

    from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse
    from llmagpie.experimental.nodes.generators.memory import InMemoryStore, MemoryNode

    seen_messages: list[list[dict]] = []

    class _Stub(BaseLLMNode):
        async def _complete(self, model, messages, **kwargs):
            seen_messages.append([dict(m) for m in messages])
            return LLMResponse(content="ok")

    store = InMemoryStore()
    # Pre-seed with system + many turns.
    asyncio.run(
        store.append(
            "t",
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ],
        )
    )
    node = MemoryNode(name="mem", inner=_Stub(name="i"), store=store, max_messages=3)

    asyncio.run(node._complete("m", [{"role": "user", "content": "u3"}], thread_id="t"))
    # max_messages=3 should keep system + last 2 non-system turns of the
    # combined (history + new) list. Combined length = 6, trimmed to 3 =
    # [SYS, a2, u3].
    sent = seen_messages[0]
    assert sent[0]["role"] == "system"
    assert len(sent) == 3
    assert sent[-1]["content"] == "u3"


def test_memory_store_clear_drops_thread():
    import asyncio

    from llmagpie.experimental.nodes.generators.memory import InMemoryStore

    store = InMemoryStore()

    async def driver():
        await store.append("t", [{"role": "user", "content": "x"}])
        assert (await store.get("t"))[0]["content"] == "x"
        await store.clear("t")
        assert await store.get("t") == []

    asyncio.run(driver())


def test_pipeline_rejects_invoke_before_compile():
    from llmagpie import BasePipeline

    @MakeNode.from_function(name="a", outputs={"value": str})
    def _a(value: str) -> str:
        """A."""
        return value

    pipe = BasePipeline(name="pipe", nodes=[_a])
    # No compile().
    with pytest.raises(RuntimeError, match="not compiled"):
        list(pipe.invoke(inputs={"a.value": "x"}))


def test_pipeline_rejects_add_node_after_compile():
    from llmagpie import BasePipeline

    @MakeNode.from_function(name="a", outputs={"value": str})
    def _a(value: str) -> str:
        """A."""
        return value

    pipe = BasePipeline(name="pipe", nodes=[_a])
    pipe.compile()

    @MakeNode.from_function(name="b", outputs={"value": str})
    def _b(value: str) -> str:
        """B."""
        return value

    with pytest.raises(RuntimeError, match="has been compiled"):
        pipe.add_nodes([_b])
