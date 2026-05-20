from __future__ import annotations

import json
import time
import uuid
from asyncio import FIRST_COMPLETED, CancelledError, get_running_loop, wait
from collections.abc import AsyncGenerator, Sequence
from typing import (
    Self,
    cast,
)

from pydantic import Field

from llmagpie.base.connectable import BaseConnectable, FunctionSchema, InternalDictState
from llmagpie.base.enum import ConnectableType, NodeRunningStatus
from llmagpie.base.utils.state import StateResponse

# opentelemetry
from llmagpie.core.opentelemetry import (
    OTEL_ENABLED,
    context,
    opentelemetry_tracer,
    trace,  # type: ignore
)

from ._aux import decompose_pipeline, make_as_task
from ._dag import SingleDAG


class _BaseTypePipeline(BaseConnectable):
    """Class that represents a pipeline of tasks.
    It inherits from BaseConnectable and implements the necessary methods to run tasks in a pipeline.
    The class includes methods for adding tasks to the pipeline, running the pipeline,
    and checking the status of the pipeline. It also includes methods
    for handling errors and cancelling the pipeline.
    """

    connectable_type: ConnectableType = ConnectableType.PIPELINE
    nodes: list[BaseConnectable] = Field(default_factory=list)
    graph: SingleDAG = Field(default_factory=lambda: SingleDAG(name=uuid.uuid4().hex))

    is_compiled: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_nodes(self.nodes)

    def compile(self, *args, **kwargs) -> Self:
        # CQJU FIXME 1009
        self._validate()
        self.graph.validate()

        # cqju: for pipeline, its internal schema corresponds to the external schema of its components

        # Generate pipeline input schema
        _dt_input, _dt_input_required = {}, []
        for node in self.graph.head_nodes:
            node_obj = self.graph.nodes[node]["_obj"]

            # Map external schema
            node_obj.func_schema.external.input.all = {
                f"{node_obj.name}.{k}": v
                for k, v in node_obj.func_schema.internal.input.all.items()
            }
            _dt_input.update(node_obj.func_schema.external.input.all)

            node_obj.func_schema.external.input.required = {
                f"{node_obj.name}.{k}" for k in node_obj.func_schema.internal.input.required
            }
            _dt_input_required += node_obj.func_schema.external.input.required

        # Generate pipeline output schema
        _dt_output: dict = {}
        for node in self.graph.tail_nodes:
            node_obj = self.graph.nodes[node]["_obj"]

            node_obj.func_schema.external.output.all = {
                f"{node_obj.name}.{k}": v
                for k, v in node_obj.func_schema.internal.output.all.items()
            }
            _dt_output.update(node_obj.func_schema.external.output.all)

        self.func_schema = FunctionSchema(
            **{
                "internal": {
                    "input": {
                        "required": _dt_input_required,
                        "all": _dt_input,
                    },
                    "output": {
                        # "required": [],
                        "all": _dt_output,
                    },
                },
            }
        )

        self.is_compiled = True
        return self

    def add_nodes(self, nodes: Sequence[BaseConnectable] | dict[str, BaseConnectable]):
        """
        Adds nodes to the pipeline.

        Args:
            nodes: A sequence or dictionary of nodes to add to the pipeline.
        """
        if self.is_compiled:
            raise RuntimeError(f"Pipeline {self.name} has been compiled!")
        if isinstance(nodes, Sequence):
            for n in nodes:
                self._add_node(n, n.name)
        elif isinstance(nodes, dict):
            for n_name, n in nodes.items():
                self._add_node(n, n_name)

    def _add_node(self, node: BaseConnectable, node_key: str):
        if self.is_compiled:
            raise RuntimeError(f"Pipeline {self.name} has been compiled!")
        if node not in self.nodes:
            self.nodes.append(node)
        # bind pipeline reference to node
        node.pipeline = self

        if node._id not in self.graph.nodes:
            self.graph.add_node(node._id, _obj=node)
            # self.binded_nodes[node_key] = node
            # self.binded_nodes[node._id] = node_key

    def add_edge(
        self,
        src_connectable: BaseConnectable,
        dest_connectable: BaseConnectable,
        src_key: list[str] | str,
        dest_key: list[str] | str,
    ):
        """Add edge."""
        if self.is_compiled:
            raise RuntimeError(f"Pipeline {self.name} has been compiled!")
        if isinstance(src_key, str):
            src_key = [src_key]
        if isinstance(dest_key, str):
            dest_key = [dest_key]

        o_schema = src_connectable.func_schema.internal.output.all
        i_schema = dest_connectable.func_schema.internal.input.all
        _in_keys, _out_keys = [], []

        for i_key, o_key in zip(dest_key, src_key, strict=False):
            if i_key not in i_schema:
                raise ValueError(f"{i_key} not in {i_schema}")
            if o_key not in o_schema:
                raise ValueError(f"{o_key} not in {o_schema}")
            o_key_schema = o_schema[o_key].get("type", "object")
            i_key_schema = i_schema[i_key].get("type", "object")
            if i_key_schema != o_key_schema:
                raise ValueError(
                    f"The schema does not align: input: {i_key}->{i_key_schema}; "
                    f"output: {o_key}->{o_key_schema}"
                )

            # bind all output keys to input
            dest_connectable._input_keys_nodes_map[i_key] = (
                dest_connectable._input_keys_nodes_map.get(i_key, [])
            )
            dest_connectable._input_keys_nodes_map[i_key].append(src_connectable._id)

            dest_connectable._input_keys_bound.add(i_key)

            _in_keys.append(i_key)
            _out_keys.append(o_key)

        dest_connectable.is_start = False
        src_connectable.is_end = False

        assert all(ele in self.graph for ele in [src_connectable._id, dest_connectable._id])

        if self.graph.has_edge(src_connectable._id, dest_connectable._id):
            # if edge already exists, just update
            edge_data = self.graph.get_edge_data(src_connectable._id, dest_connectable._id)
            for key in self.graph[src_connectable._id][dest_connectable._id]:
                if key in edge_data:
                    if key == "_output_keys":
                        self.graph[src_connectable._id][dest_connectable._id][key] += src_key
                    if key == "_input_keys":
                        self.graph[src_connectable._id][dest_connectable._id][key] += dest_key
        else:
            self.graph.add_edge(
                u_of_edge=src_connectable._id,
                v_of_edge=dest_connectable._id,
                ##
                _output_keys=_out_keys,
                _input_keys=_in_keys,
            )

    def _flatten_history_object_store(self, session_id: str) -> InternalDictState:
        res = {
            session_id: {
                ".".join(_key.split(".")[1:]): _value
                for _key, _value in self.input_state.get(session_id, {}).items()
            }
        }
        return cast(InternalDictState, res)

    def _collect_head_tasks(self, session_id: str, inputs: dict, root_nodes: list[BaseConnectable]):
        try:
            _root_node_input_schema = set(self.func_schema.internal.input.all.keys())
            _root_node_required_input = set(self.func_schema.internal.input.required)

            given = set(inputs.keys())
            if not (
                _root_node_required_input.issubset(given)
                and given.issubset(_root_node_input_schema)
            ):
                raise ValueError(
                    "Required input parameters are not fully bound, or unknown keys "
                    f"bound. given={given}, required={_root_node_required_input}, "
                    f"schema={_root_node_input_schema}"
                )

            _task_dict = dict()

            for _child in root_nodes:
                assert _child.is_start
                if _child.input_state == {} and self.input_state.get(session_id, []):
                    try:
                        _child.input_state = self._flatten_history_object_store(session_id)
                    except Exception as exc:
                        # self.logger.warning(f"{_child} is the head node of the session.")
                        self._error_callback(session_id, exc)

                # TODO: validate at compile() that input parameter names are unique
                # across head nodes; the current `{prefix}.{key}` split below assumes it.
                _inputs = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in inputs.items()
                    if ".".join(k.split(".")[1:]) in _child.func_schema.internal.input.all
                }
                _inputs = _child.precheck(session_id=session_id, inputs=_inputs)
                if _inputs:
                    _iterator_target = _child.async_event_on_execution(
                        session_id=session_id,
                        inputs=_inputs,
                    )
                    _task_dict.update(
                        {
                            _iterator_target: {
                                "node_id": _child._id,
                                "node_name": _child.name,
                                "iterator": _iterator_target,
                            }
                        }
                    )

            return _task_dict
        except Exception as exc:
            self._error_callback(session_id, exc)

    def _collect_children_tasks(
        self,
        session_id: str,
        output_values_internal: dict,
        parent: BaseConnectable,
    ):
        # check if parent had valid outputs; if not, no emission
        if not any(x is not None for x in output_values_internal.values()):
            return {}
        # try:
        #     assert any(x is not None for x in output_values_internal.values()), \
        #         f"{parent.name}: No parameter is filled; all are None."
        # except AssertionError as exc:
        #     self._error_callback(session_id, CancelledError(exc))

        try:
            # EMIT TO CHILDREN of the node
            iterator_dt = dict()

            for child_id, edge_info_dict in self.graph.succ[parent._id].items():
                # NODE and EDGE: get child and its mapping keys
                child, _input_keys, _output_keys = (
                    self.graph.nodes[child_id]["_obj"],
                    edge_info_dict["_input_keys"],
                    edge_info_dict["_output_keys"],
                )
                # emit value to children: change the name from output into input
                if child.input_state.get(session_id, None) is None:
                    child.input_state[session_id] = child.input_state.get(session_id, {})

                _input_values_internal = {
                    s: output_values_internal[d]
                    for s, d in zip(_input_keys, _output_keys, strict=False)
                    if output_values_internal.get(d, None)
                }
                if _input_values_internal != {}:
                    for _key, _value in _input_values_internal.items():
                        child.input_state[session_id][_key] = child.input_state.get(
                            session_id, {}
                        ).get(_key, [])
                        child.input_state[session_id][_key] += [
                            {
                                parent._id: {
                                    "_timestamp": time.time(),
                                    "value": _value,
                                }
                            }
                        ]

                    _inputs = child.precheck(session_id=session_id)
                    if _inputs:
                        self.logger.debug("%s -> emitted to -> %s", parent.name, child.name)
                        iterator_target = child.async_event_on_execution(
                            session_id=session_id,
                            inputs=_inputs,
                        )

                        iterator_dt.update(
                            {
                                iterator_target: {
                                    "node_id": child._id,
                                    "node_name": child.name,
                                    "iterator": iterator_target,
                                    # "task"
                                }
                            }
                        )
                    else:
                        self.logger.debug(
                            "%s -> emitted to -> %s but NOT executed", parent.name, child.name
                        )
                else:
                    self.logger.debug(
                        "%s -> NOT emitted to -> %s: Input empty", parent.name, child.name
                    )
            return iterator_dt

        except CancelledError as exc:
            self._error_callback(
                session_id, Exception(f"{parent.name}: Task async_emit has been cancelled. {exc}")
            )
        except Exception as exc:
            self._error_callback(session_id, exc)

    def _callback(self, session_id):
        # after execution, self input object store should be cleaned

        # collect from its included components
        _output_values = {}
        for _node_id in self.graph.nodes:
            node = self.graph.nodes[_node_id]["_obj"]
            _output_values[node.name] = node.output_state.get(session_id, None)

        self.output_state[session_id] = self.output_state.get(session_id, [])
        self.output_state[session_id].append(
            {"_timestamp": time.time(), "_type": self.connectable_type, "value": _output_values}
        )

        self.output_history_state[session_id] = self.output_history_state.get(session_id, [])
        self.output_history_state[session_id].append(
            {"_timestamp": time.time(), "_type": self.connectable_type, "value": _output_values}
        )

        return _output_values

    async def async_event_on_execution(
        self, inputs: dict | None, session_id: str, **kwargs
    ) -> AsyncGenerator:
        """EXECUTION when the node is triggered."""
        try:
            if not self.is_compiled:
                raise RuntimeError(
                    f"Pipeline {self.name} is not compiled yet; please compile it "
                    "first using `pipe.compile()`."
                )
            aioloop = get_running_loop()
            self._running_status = NodeRunningStatus.RUNNING
            if inputs:
                if OTEL_ENABLED:
                    span = opentelemetry_tracer._tracer.start_span(self.name)
                    span.set_attributes(
                        {
                            "input.value": json.dumps(inputs),
                        }
                    )
                    span.set_attributes(
                        {
                            "openinference.span.kind": "CHAIN",
                        }
                    )
                    # Creates a Context object with parent set as current span
                    ctx = trace.set_span_in_context(span)
                    # Set as the implicit current context. The matching detach
                    # is disabled below; see the FIXME at the end of this method.
                    context.attach(ctx)

                root_nodes = [
                    self.graph.nodes[_node_id]["_obj"] for _node_id in self.graph.head_nodes
                ]
                task_dict = self._collect_head_tasks(session_id, inputs, root_nodes)
                # compile upon running
                iterator_dict = {
                    asset["iterator"]: {
                        "node_id": asset["node_id"],
                        "node_name": asset["node_name"],
                        "iterator": asset["iterator"],
                        "task": make_as_task(asset["iterator"], aioloop),
                    }
                    for _, asset in task_dict.items()
                }

                while iterator_dict:
                    done_tasks, _pending = await wait(
                        [ele["task"] for ele in iterator_dict.values()], return_when=FIRST_COMPLETED
                    )
                    for done_task in done_tasks:
                        node_name, node_id, iterator = next(
                            (t["node_name"], t["node_id"], t["iterator"])
                            for it, t in iterator_dict.items()
                            if t["task"] == done_task
                        )

                        try:
                            response: StateResponse = done_task.result()
                            if response:
                                if OTEL_ENABLED:
                                    # IMPORTANT: save context before yield!
                                    current_ctx = context.get_current()
                                _output_values: dict = response.value
                                _parent = response.node
                                # This yield to the final output
                                yield StateResponse(
                                    timestamp=time.time(),
                                    type=_parent.connectable_type,
                                    value=_output_values,
                                    node=_parent,
                                )
                                if OTEL_ENABLED:
                                    span.set_attributes(
                                        {
                                            f"component_output.value.{response.node.name}": json.dumps(
                                                _output_values
                                            ),
                                        }
                                    )
                                    context.attach(current_ctx)
                            del response

                        except StopAsyncIteration:
                            self.logger.debug("%s: StopAsyncIteration", node_name)
                            del iterator_dict[iterator]

                            _parent = self.graph.nodes[node_id]["_obj"]

                            if isinstance(_parent.output_history_state[session_id], list):
                                _most_recent_output_values = _parent.output_history_state[
                                    session_id
                                ].pop(-1)["value"]
                            else:
                                raise ValueError(
                                    f"output_history_state type is wrong: {type(_parent.output_history_state[session_id])}"
                                ) from None

                            if _parent.connectable_type == ConnectableType.PIPELINE:
                                _most_recent_output_values = decompose_pipeline(
                                    _most_recent_output_values
                                )
                            else:
                                pass

                            if not _parent.is_end:
                                # collect the infomation for children nodes
                                _c_tasks_dict: dict | None = self._collect_children_tasks(
                                    session_id=session_id,
                                    output_values_internal=_most_recent_output_values,
                                    parent=_parent,
                                )
                                if _c_tasks_dict:
                                    for _, _asset in _c_tasks_dict.items():
                                        iterator_dict.update(
                                            {
                                                _asset["iterator"]: {
                                                    "node_id": _asset["node_id"],
                                                    "node_name": _asset["node_name"],
                                                    "iterator": _asset["iterator"],
                                                    "task": make_as_task(
                                                        _asset["iterator"], aioloop
                                                    ),
                                                }
                                            }
                                        )
                        except Exception as exc:
                            self._error_callback(session_id, exc)
                        else:
                            # The iterator hasn't exhausted or errored out.
                            # Queue the next inspection.
                            iterator_dict[iterator] = {
                                "node_id": node_id,
                                "node_name": node_name,
                                "iterator": iterator,
                                "task": make_as_task(iterator, aioloop),
                            }
                # Finally, callback function here
                _output_values = self._callback(session_id)

                if OTEL_ENABLED:
                    span.set_attributes(
                        {
                            "output.value": json.dumps(_output_values),
                        }
                    )
                    span.end()

                # Don't forget to detach or parent will remain the parent above this call stack
                # FIXME: context.detach(token) raises "ContextVar token was created in a different Context"
                # when the attach happens inside an async generator. Re-enable once the token is
                # released in the same context it was created in (likely via a try/finally around
                # the yields, or by switching to span context-management instead of attach/detach).
                # context.detach(token)

            self._running_status = NodeRunningStatus.INACTIVE
        except Exception as exc:
            self._error_callback(session_id, exc)
        finally:
            self.count_visited += 1


class BasePipeline(_BaseTypePipeline):
    def _validate(self):
        self._validate_root_nodes()
        # migrate heads and tails check from node level
        # Check in-degree and out-degree of nodes (on pipeline)
        for _id in self.graph.nodes:
            _node = self.graph.nodes[_id]["_obj"]
            in_ok = (self.graph.in_degree(_id) == 0 and _node.is_start is True) or (
                self.graph.in_degree(_id) != 0 and _node.is_start is not True
            )
            if not in_ok:
                raise ValueError(
                    f"{self.__class__.__name__} pipeline in-degree is wrong for node {_node.name}."
                )
            out_ok = (self.graph.out_degree(_id) == 0 and _node.is_end is True) or (
                self.graph.out_degree(_id) != 0 and _node.is_end is not True
            )
            if not out_ok:
                raise ValueError(
                    f"{self.__class__.__name__} pipeline out-degree is wrong for node {_node.name}."
                )

    def _validate_root_nodes(self):
        if len(self.graph.head_nodes) < 1:
            raise ValueError(f"At least one root node is required in {self.__class__.__name__}.")
