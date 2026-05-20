"""Targeted unit tests for behaviors that regressed in the past or are
easy to break without noticing."""

import asyncio

import pytest

from llmagpie.base.connectable import BaseConnectable
from llmagpie.base.enum import ConnectableType
from llmagpie.base.logging.logging_wrapper import log_output
from llmagpie.base.node import BaseNode, MakeNode


class _Concrete(BaseConnectable):
    """Minimal concrete BaseConnectable for state-isolation tests."""
    connectable_type: ConnectableType = ConnectableType.BASENODE

    def _validate(self):
        return True

    async def async_event_on_execution(self, inputs, session_id, **kwargs):
        if False:
            yield


def test_input_keys_are_per_instance():
    """`_input_keys_binded` / `_input_keys_nodes_map` used to be declared
    as bare class attributes — every BaseConnectable instance shared the
    same set/dict. They are now PrivateAttr-backed per-instance state."""
    a = _Concrete(name="a")
    b = _Concrete(name="b")

    a._input_keys_binded.add("x")
    a._input_keys_nodes_map["x"] = ["node-a"]

    assert b._input_keys_binded == set()
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


def test_make_node_from_function_runs():
    @MakeNode.from_function(name="echo", outputs={"value": str})
    def _echo(value: str) -> str:
        """Echo back the input."""
        return value

    assert isinstance(_echo, BaseNode)
    assert _echo.run(value="hi") == {"value": "hi"}
