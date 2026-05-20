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
