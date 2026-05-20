"""Targeted tests that fill in coverage gaps across the framework's
core modules. Each test names the file:lines it's exercising so a
future drop in coverage can be diagnosed quickly.

NOTE: deliberately NO `from __future__ import annotations` here —
several tests want the actual ``Annotated`` form at runtime so
``typing.get_origin`` can detect it inside `_schema.py`.
"""

import asyncio
import os
import tempfile

import pytest
from llmagpie.base.node import BaseNode, MakeNode

# ---------------------------------------------------------------------------
# post_run.py: Generator / AsyncGenerator / tuple / non-dict scalar paths.
# ---------------------------------------------------------------------------


def test_post_run_handles_generator_of_dicts():
    from llmagpie.base.node._post_run import post_run
    from pydantic import BaseModel

    class M(BaseModel):
        v: int

    def gen():
        yield {"v": 1}
        yield {"v": 2}

    out = list(post_run(gen(), M))
    assert out == [{"v": 1}, {"v": 2}]


def test_post_run_handles_generator_of_tuples_and_scalars():
    from llmagpie.base.node._post_run import post_run
    from pydantic import BaseModel

    class M(BaseModel):
        v: int

    def gen():
        yield (10,)
        yield 20  # scalar — folded into the single field

    assert list(post_run(gen(), M)) == [{"v": 10}, {"v": 20}]


def test_post_run_handles_async_generator():
    from llmagpie.base.node._post_run import post_run
    from pydantic import BaseModel

    class M(BaseModel):
        v: int

    async def agen():
        yield {"v": 1}
        yield (2,)

    async def driver():
        return [x async for x in post_run(agen(), M)]

    assert asyncio.run(driver()) == [{"v": 1}, {"v": 2}]


def test_post_run_tuple_input():
    from llmagpie.base.node._post_run import post_run
    from pydantic import BaseModel

    class M(BaseModel):
        v: int

    assert post_run((42,), M) == {"v": 42}


def test_post_run_scalar_fallback_path():
    from llmagpie.base.node._post_run import post_run
    from pydantic import BaseModel

    class M(BaseModel):
        v: int

    # A scalar that's not dict/tuple/Generator/AsyncGenerator — hits the
    # last try block.
    assert post_run(7, M) == {"v": 7}


# ---------------------------------------------------------------------------
# BaseNode: run / async_run / stream / async_stream variants + the
# error branch in _async_stream for unsupported return shapes.
# ---------------------------------------------------------------------------


def _make_dict_node():
    @MakeNode.from_class(func_name="async_call", outputs={"v": str})
    class _D(BaseNode):
        async def async_call(self, name: str):
            return {"v": name.upper()}

    return _D(name="d")


def test_basenode_async_run_returns_last_value():
    n = _make_dict_node()
    assert asyncio.run(n.async_run(name="hi")) == {"v": "HI"}


def test_basenode_async_stream_returns_an_async_generator():
    n = _make_dict_node()

    async def driver():
        out = []
        ag = await n.async_stream(name="hi")
        async for v in ag:
            out.append(v)
        return out

    assert asyncio.run(driver()) == [{"v": "HI"}]


def test_basenode_run_drives_event_loop_sync():
    n = _make_dict_node()
    assert n.run(name="hello") == {"v": "HELLO"}


def test_basenode_unsupported_async_call_return_raises():
    # The "must return a dict" guard in `_async_stream` is only reachable
    # when async_call_ returns something that isn't an Awaitable, Generator,
    # AsyncGenerator, or dict. MakeNode normally wraps the user's call in
    # an awaitable that returns post-validated dicts, so the only way to
    # hit this branch is by directly assigning a non-conforming callable.
    n = _make_dict_node()

    def _bogus(**kw):
        return 42  # not Awaitable, not Generator, not AsyncGenerator, not dict

    object.__setattr__(n, "async_call_", _bogus)
    with pytest.raises(TypeError, match="must return a dict"):
        n.run(name="x")


# ---------------------------------------------------------------------------
# tools.py: schema generation, repr/str, and the dispatch-failure branch.
# ---------------------------------------------------------------------------


def test_tools_node_schema_and_repr():
    from llmagpie.base.tools import ToolsNode

    @MakeNode.from_function(name="upper", outputs={"value": str})
    def _upper(value: str) -> str:
        """Uppercase the input."""
        return value.upper()

    t = ToolsNode(name="t", tools=[_upper])

    schema = t._generate_openai_schema()
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "upper"
    # `__repr__` / `__str__` go through tools_with_mapping.
    assert "upper" in repr(t)
    assert "upper" in str(t)


def test_tools_node_dispatches_unknown_tool_to_error():
    from llmagpie.base.tools import ToolsNode

    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo."""
        return value

    t = ToolsNode(name="t", tools=[_echo])
    out = t.run(
        tool_calls_list=[
            {"function": {"name": "no-such-tool", "arguments": "{}"}},
        ]
    )
    result = out["tool_calls_list"][0]
    # Unknown tool → KeyError caught, error populated, output stays None.
    assert result["output"] is None
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# _schema.py: Annotated handling + varargs/varkw rejection in
# create_schema_from_function.
# ---------------------------------------------------------------------------


def test_create_schema_from_function_supports_annotated():
    from typing import Annotated

    from llmagpie.base.node._schema import create_schema_from_function

    def f(x: Annotated[int, "the x"]) -> int:
        return x

    model = create_schema_from_function(f)
    props = model.model_json_schema()["properties"]
    # The Annotated metadata is surfaced as `description`.
    assert props["x"]["description"] == "the x"


def test_create_schema_rejects_var_args_or_var_keyword():
    from llmagpie.base.node._schema import create_schema_from_function

    def f_args(*args): ...
    def f_kw(**kw): ...

    with pytest.raises(ValueError, match="not allowed"):
        create_schema_from_function(f_args)
    with pytest.raises(ValueError, match="not allowed"):
        create_schema_from_function(f_kw)


# ---------------------------------------------------------------------------
# node_wrapper.py: MakeNode.from_function failure paths.
# ---------------------------------------------------------------------------


def test_make_node_from_function_rejects_missing_docstring():
    with pytest.raises(ValueError, match="Tools does not have description"):

        @MakeNode.from_function(name="bad", outputs={"v": str})
        def _no_doc(v: str) -> str:
            return v


def test_make_node_from_function_rejects_varargs():
    with pytest.raises(ValueError, match="kwargs"):

        @MakeNode.from_function(name="bad", outputs={"v": str})
        def _varargs(*args) -> str:
            """no good."""
            return ""


# ---------------------------------------------------------------------------
# _dag.py: head/tail validation raise paths + the disabled
# cycle-detection helper (called directly so the asserts fire).
# ---------------------------------------------------------------------------


def test_singledag_validate_heads_and_tails_raises():
    from llmagpie.base.pipeline._dag import SingleDAG

    g = SingleDAG()  # empty
    with pytest.raises(ValueError, match="no head"):
        g._validate_heads_and_tails()


def test_singledag_validate_edges_circular_accepts_acyclic():
    from llmagpie.base.pipeline._dag import SingleDAG

    g = SingleDAG()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b")
    # Should not raise: graph is acyclic, no cycles.
    g._validate_edges_circular()


# ---------------------------------------------------------------------------
# retry.py: parameter-validation error paths.
# ---------------------------------------------------------------------------


def test_with_retry_rejects_invalid_params():
    from llmagpie.base.utils.retry import with_retry

    with pytest.raises(ValueError, match="max_attempts"):
        with_retry(max_attempts=0)
    with pytest.raises(ValueError, match="backoff_base"):
        with_retry(backoff_base=-1.0)


# ---------------------------------------------------------------------------
# state.py: StateResponse.to_dict(recursive=True) goes through model_dump.
# ---------------------------------------------------------------------------


def test_state_response_to_dict_recursive():
    from llmagpie.base.connectable import BaseConnectable
    from llmagpie.base.enum import ConnectableType
    from llmagpie.base.utils.state import StateResponse

    class _Concrete(BaseConnectable):
        connectable_type: ConnectableType = ConnectableType.BASENODE

        def _validate(self):
            return True

        async def async_event_on_execution(self, inputs, session_id, **kwargs):
            if False:
                yield

    node = _Concrete(name="n")
    sr = StateResponse(timestamp=1.0, type=ConnectableType.BASENODE, value={"k": "v"}, node=node)
    flat = sr.to_dict(recursive=False)
    deep = sr.to_dict(recursive=True)
    assert flat["value"] == {"k": "v"}
    assert deep["value"] == {"k": "v"}


# ---------------------------------------------------------------------------
# thread.py: AsyncThread rejects unsupported coro types.
# ---------------------------------------------------------------------------


def test_async_thread_rejects_non_awaitable_non_asyncgen():
    import asyncio as _aio

    from llmagpie.base.utils.thread import AsyncThread

    loop = _aio.new_event_loop()
    try:
        t = AsyncThread(coro="not-a-coroutine", loop=loop)  # type: ignore[arg-type]
        t.start()
        t.join()
        # `run()` caught the TypeError and stored it on `result`.
        assert isinstance(t.result, TypeError)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# logging_wrapper.py: log_output should treat module-level functions as
# non-methods (the `_is_method` False branch).
# ---------------------------------------------------------------------------


def test_log_output_on_sync_method_logs_output():
    """Cover the is_method=True branch in the sync wrapper (line 38)."""
    from llmagpie.base.logging.logging_wrapper import log_output

    class Holder:
        name = "the-holder"

        @log_output
        def make(self, x):
            return f"got {x}"

    assert Holder().make(7) == "got 7"


def test_log_output_on_async_method_logs_output():
    """Cover the is_method=True branch in the async wrapper (line 29)."""
    import asyncio as _aio

    from llmagpie.base.logging.logging_wrapper import log_output

    class Holder:
        name = "the-holder"

        @log_output
        async def make(self, x):
            return f"async-got {x}"

    assert _aio.run(Holder().make(7)) == "async-got 7"


def test_log_output_handles_callable_without_argspec():
    """Cover the IndexError/TypeError fallback in `_is_method` (lines 12-13).
    A builtin like `len` raises TypeError from getfullargspec; the
    decorator should treat it as non-method and just pass through."""
    from llmagpie.base.logging.logging_wrapper import _is_method

    assert _is_method(len) is False  # getfullargspec(len) raises TypeError


def test_log_output_non_method_does_not_log_output():
    from llmagpie.base.logging.logging_wrapper import log_output

    @log_output
    def plain(x):
        return x + 1

    assert plain(2) == 3  # just returns; no exception trying to read .name


# ---------------------------------------------------------------------------
# logging.py: file-handler path activated via LOG_DIR env var.
# ---------------------------------------------------------------------------


def test_get_or_create_logger_uses_file_handler_when_log_dir_set(monkeypatch):
    """Exercise the LOG_DIR file-handler branch of get_or_create_logger.

    We can't reliably assert the resulting logger HAS the file handler
    (the DefaultLogger helper skips handler attachment when the logger
    inherits handlers from root — which pytest's setup tends to cause).
    Instead, assert the side effect: the log directory is created and
    `info.log` exists or the call returns a logger without raising.
    """
    from llmagpie.base.logging.logging import get_or_create_logger

    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "logs-subdir")  # not yet created
        monkeypatch.setenv("LOG_DIR", target)
        log = get_or_create_logger(logger_name=f"file-handler-test-{os.getpid()}")
        assert log is not None
        # The function calls os.makedirs(log_path, exist_ok=True).
        assert os.path.isdir(target)


# ---------------------------------------------------------------------------
# batch.py: max_concurrency semaphore branch.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BaseNode._async_stream: Generator and AsyncGenerator return paths.
# ---------------------------------------------------------------------------


def test_basenode_async_call_returns_sync_generator():
    """Cover the `isinstance(res, Generator)` branch in _async_stream."""
    # Bypass MakeNode so the raw return type reaches _async_stream.
    n = _make_dict_node()

    def _gen_call(**kw):
        def g():
            yield {"v": "g1"}
            yield {"v": "g2"}

        return g()

    object.__setattr__(n, "async_call_", _gen_call)
    out = list(n.stream(name="x"))
    assert out == [{"v": "g1"}, {"v": "g2"}]


def test_basenode_async_call_returns_async_generator():
    """Cover the `isinstance(res, AsyncGenerator)` branch."""
    n = _make_dict_node()

    def _agen_call(**kw):
        async def ag():
            yield {"v": "a1"}
            yield {"v": "a2"}

        return ag()

    object.__setattr__(n, "async_call_", _agen_call)
    out = list(n.stream(name="x"))
    assert out == [{"v": "a1"}, {"v": "a2"}]


# ---------------------------------------------------------------------------
# _post_run.py: scalar-fallback TypeError path (last try/except).
# ---------------------------------------------------------------------------


def test_post_run_raises_on_unconvertible_scalar():
    from llmagpie.base.node._post_run import post_run
    from pydantic import BaseModel

    class Strict(BaseModel):
        v: int  # int field; passing a non-coercible value triggers ValidationError

    with pytest.raises(TypeError, match="Result type is wrong"):
        post_run(object(), Strict)


# ---------------------------------------------------------------------------
# retry.py: with_fallback composed with with_retry (the "happy fallback" path).
# ---------------------------------------------------------------------------


def test_with_fallback_passthrough_when_primary_succeeds():
    """Cover the no-exception branch of with_fallback."""
    import asyncio as _aio

    from llmagpie.base.utils.retry import with_fallback

    async def _fb(x):
        return f"fallback({x})"

    @with_fallback(_fb)
    async def primary(x):
        return f"primary({x})"

    assert _aio.run(primary("ok")) == "primary(ok)"


# ---------------------------------------------------------------------------
# routing.py: multi_switch with selector via the `dest_key` kwarg branch.
# ---------------------------------------------------------------------------


def test_multi_switch_no_match_routes_nothing():
    from llmagpie import BasePipeline
    from llmagpie.base.utils.routing import multi_switch

    @MakeNode.from_class(func_name="async_call", outputs={"k": str})
    class _Src(BaseNode):
        async def async_call(self, k: str):
            return {"k": k}

    @MakeNode.from_class(func_name="async_call", outputs={"out": str})
    class _A(BaseNode):
        async def async_call(self, k: str):
            return {"out": "A"}

    src, a = _Src(name="src"), _A(name="a")
    pipe = BasePipeline(name="p", nodes=[src, a])
    multi_switch(pipe, src, src_key="k", dest_key="k", branches={"alpha": a})
    pipe.compile()
    # Emit a value that doesn't match any branch — a's cond_func returns
    # False, so it doesn't fire.
    finals = [s.value for s in pipe.invoke(inputs={"src.k": "beta"})]
    assert {"out": "A"} not in finals


# ---------------------------------------------------------------------------
# __init__.py: the PackageNotFoundError branch (lines 9-11).
# ---------------------------------------------------------------------------


def test_package_metadata_fallback_when_not_installed(monkeypatch):
    """Exercise the `except metadata.PackageNotFoundError` branch."""
    import importlib

    # Force the metadata.version call to raise PackageNotFoundError so the
    # except branch executes.
    from importlib import metadata as _metadata

    import llmagpie

    def _boom(*_a, **_kw):
        raise _metadata.PackageNotFoundError("forced")

    monkeypatch.setattr(_metadata, "version", _boom)
    importlib.reload(llmagpie)
    assert llmagpie.__version__ == ""
    # Reload again normally so other tests see a sane state.
    monkeypatch.undo()
    importlib.reload(llmagpie)


# ---------------------------------------------------------------------------
# node/_base.py: thread_mode branch of `stream` — when an event loop is
# already running, the framework dispatches via a worker thread.
# ---------------------------------------------------------------------------


def test_basenode_stream_under_running_loop_uses_thread_mode():
    """Calling .stream() from inside an asyncio task forces the
    'a loop is already running' branch (lines 92-100 of node/_base.py)."""

    n = _make_dict_node()

    async def driver():
        # Run the sync .stream() in the default executor so it's called
        # while THIS loop is running.
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: list(n.stream(name="hi"))
        )

    out = asyncio.run(driver())
    assert out == [{"v": "HI"}]


# ---------------------------------------------------------------------------
# BaseConnectDisposable: __lshift__ / __rshift__ — exercised via the
# `(src >> "out") >> ("in" >> dest)` operator form. These methods (lines
# 107-114 of connectable.py) get hit by wiring two disposables together.
# ---------------------------------------------------------------------------


def test_connect_disposable_operators_wire_edges():
    from llmagpie import BasePipeline

    @MakeNode.from_class(func_name="async_call", outputs={"v": str})
    class _N(BaseNode):
        async def async_call(self, v: str):
            return {"v": v}

    a, b = _N(name="a"), _N(name="b")
    pipe = BasePipeline(name="p", nodes=[a, b])

    # (a >> "v") returns a BaseConnectDisposable; ("v" >> b) returns
    # another; the outer >> calls BaseConnectDisposable.__rshift__.
    (a >> "v") >> ("v" >> b)
    pipe.compile()

    finals = [s.value for s in pipe.invoke(inputs={"a.v": "ok"})]
    assert {"v": "ok"} in finals


# ---------------------------------------------------------------------------
# logging.py: formatTime TypeError fallback (line 42-45). Triggered by
# pytz.timezone datetime that doesn't support `timespec` on older
# Python — we force it by mocking isoformat.
# ---------------------------------------------------------------------------


def test_custom_formatter_handles_timespec_typeerror():
    import logging as _logging

    from llmagpie.base.logging.logging import CustomFormatter

    fmt = CustomFormatter("%(message)s")

    # A record with no datefmt argument and a datetime whose .isoformat
    # raises TypeError when given timespec — simulate by patching.
    record = _logging.LogRecord(
        name="t",
        level=_logging.INFO,
        pathname="",
        lineno=0,
        msg="x",
        args=(),
        exc_info=None,
    )
    real_converter = fmt.converter

    def _converter(ts):
        dt = real_converter(ts)

        class _NoTimespec:
            def isoformat(self, timespec=None):
                if timespec is not None:
                    raise TypeError("no timespec support")
                return dt.isoformat()

            def strftime(self, fmt):
                return dt.strftime(fmt)

        return _NoTimespec()

    fmt.converter = _converter  # type: ignore[assignment]
    formatted = fmt.formatTime(record, datefmt=None)
    # Confirms the TypeError fallback returned something (the
    # isoformat-without-timespec path).
    assert isinstance(formatted, str) and formatted


# ---------------------------------------------------------------------------
# node_wrapper.py: module-level `from_class` / `from_function` delegators.
# ---------------------------------------------------------------------------


def test_module_level_from_class_and_from_function_delegate():
    from llmagpie.base.node.node_wrapper import from_class, from_function

    @from_function(name="echo", outputs={"v": str})
    def _echo(v: str) -> str:
        """Echo."""
        return v

    assert _echo.run(v="ok") == {"v": "ok"}

    class _Cls:
        async def go(self, v: str):
            return {"v": v}

    Wrapped = from_class(_Cls, func_name="go", outputs={"v": str})
    assert Wrapped is _Cls  # the decorator returns the original class
    assert hasattr(_Cls, "async_call_")


def test_batch_invoke_honors_max_concurrency():
    from llmagpie import BasePipeline
    from llmagpie.base.utils.batch import batch_invoke

    @MakeNode.from_function(name="id", outputs={"v": str})
    def _id(v: str) -> str:
        """identity."""
        return v

    pipe = BasePipeline(name="p", nodes=[_id])
    pipe.compile()
    results = batch_invoke(
        pipe,
        [{"id.v": "a"}, {"id.v": "b"}, {"id.v": "c"}],
        max_concurrency=1,
    )
    finals = [r[-1].value for r in results]
    assert finals == [{"v": "a"}, {"v": "b"}, {"v": "c"}]
