from __future__ import annotations
from abc import abstractmethod
import uuid
import time
import json
import os

from asyncio import FIRST_COMPLETED, Task, create_task, wait, CancelledError
from opentelemetry import trace, context
from pydantic import BaseModel, Field
from pydantic._internal._model_construction import ModelMetaclass
from inspect import isawaitable

from llmagpie.core.dag import SingleDAG
from llmagpie.core.connectable import BaseConnectable, FunctionSchema
from llmagpie.core.nodes import BaseNode
from llmagpie.core.nodes._base import BaseNodeDisposable
from llmagpie.core.logging import get_or_create_logger
# from llmagpie.experimental.merge_iterators import merge_iterators
from llmagpie.experimental.opentelemetry import opentelemetry_tracer, OTEL_ENABLED

from typing import (
    AsyncGenerator, Collection, TypeVar, Any,
    Sequence, Dict, Union, Optional, List, Callable,
    Self
)

from ._aux import make_as_task, decompose_pipeline


class BasePipelineMixin(BaseConnectable):
    connectable_type: str = "Pipeline"
    nodes: List[BaseConnectable] = []
    graph: SingleDAG = Field(default_factory=lambda: SingleDAG(name=uuid.uuid4().hex))
    is_compiled: bool = False
    
    def __init__(
        self,
        # nodes: Union[Dict[str, BaseConnectable], Sequence[BaseConnectable]],
        *args,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.add_nodes(self.nodes)

    def compile(self, *args, **kwargs) -> "Self":
        # CQJU FIXME 1009
        self._validate()
        self.graph.validate()
        
        # cqju: for pipeline, its internal schema corresponds to the external schema of its components 
        _dt_input, _dt_input_required, _dt_output = {}, [], {}
        for node in self.graph.head_nodes:
            node_obj = self.graph.nodes[node]["_obj"]
            
            node_obj.func_schema.external.input.all = {
                f"{node_obj.name}.{k}": v for k, v in node_obj.func_schema.internal.input.all.items()
            }
            _dt_input.update(node_obj.func_schema.external.input.all)
            
         
        for node in self.graph.tail_nodes:
            node_obj = self.graph.nodes[node]["_obj"]
            
            node_obj.func_schema.external.output.all = {
                f"{node_obj.name}.{k}": v for k, v in node_obj.func_schema.internal.output.all.items()
            }
            _dt_output.update(node_obj.func_schema.external.output.all)
        
        for node in self.graph.head_nodes:
            node_obj = self.graph.nodes[node]["_obj"]
            node_obj.func_schema.external.input.required = {
                f"{node_obj.name}.{k}" for k in node_obj.func_schema.internal.input.required
            }
            _dt_input_required += (node_obj.func_schema.external.input.required)
        
        self.func_schema = FunctionSchema(**{
            "internal": {
                "input": {
                    "required": _dt_input_required,
                    "all": _dt_input,
                },
                "output": {
                    "required": [],
                    "all": _dt_output,
                },
            },
            "external": {
                "input": {
                    "required": [],
                    "all": {},
                },
                "output": {
                    "required": [],
                    "all": {},
                },
            },
        })
        
        self.is_compiled = True

        return self

    def add_nodes(self, nodes: Union[Sequence[BaseConnectable], Dict[str, BaseConnectable]]):
        assert self.is_compiled is False, f"Pipeline {self.name} has been compiled!"
        """Add nodes."""
        if isinstance(nodes, Sequence):
            for n in nodes:
                self._add_node(n, n.name)
        elif isinstance(nodes, Dict):
            for n_name, n in nodes.items():
                self._add_node(n, n_name)

    def _add_node(self, node: BaseConnectable, node_key: str):
        assert self.is_compiled is False, f"Pipeline {self.name} has been compiled!"
        # bind pipeline reference to node
        node.pipeline = self

        if node._id not in self.graph.nodes:
            self.graph.add_node(
                node._id,
                _obj=node,
            )
            # self.binded_nodes[node_key] = node
            # self.binded_nodes[node._id] = node_key

    def add_edge(
        self,
        src_node: Union[BaseConnectable, BaseNodeDisposable],
        dest_node: Union[BaseConnectable, BaseNodeDisposable],
        src_key: Optional[Union[List[str], str]] = None,
        dest_key: Optional[Union[List[str], str]] = None
    ):
        """Add edge.
        """
        assert self.is_compiled is False, f"Pipeline {self.name} has been compiled!"
        if isinstance(src_node, BaseNodeDisposable) and isinstance(dest_node, BaseNodeDisposable):
            # src_node >> dest_node
            src_node._set_edge(dest_node, upstream=False)

        elif isinstance(src_node, BaseConnectable) and isinstance(dest_node, BaseConnectable):
            if isinstance(src_key, str):
                src_key = [src_key]
            if isinstance(dest_key, str):
                dest_key = [dest_key]
            BaseNodeDisposable(connectable=src_node, out_keys=src_key)._set_edge(
                BaseNodeDisposable(connectable=dest_node, in_keys=dest_key),
                upstream=False
            )
        else:
            raise TypeError("type is wrong")

    # async def _execute(self, **kwargs):
    #     ...
    def _flatten_history_object_store(self, session_id: str) -> Dict[str, Dict]:
        res = {
            session_id: {
                ".".join(_key.split('.')[1:]): _value for _key, _value in self.history_state.get(session_id, {}).items()
            }
        }
        return res
    
    def _collect_head_tasks(
        self,
        session_id: str,
        inputs: Dict,
        root_nodes: List[BaseConnectable]
    ):
        try:
            assert self.is_compiled, f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."
            _root_node_input_schema = set( self.func_schema.internal.input.all.keys() )
            _root_node_required_input = set( self.func_schema.internal.input.required )
            
            assert set(_root_node_required_input).issubset(set(inputs.keys())) \
                and set(inputs.keys()).issubset(_root_node_input_schema), "Required inputs parameters are not fully bound. Or unknown keys bound."

            _task_dict = dict()

            for _child in root_nodes:
                assert _child.is_start
                if _child.history_state == {} and self.history_state.get(session_id, []):
                    print("***self: ", self, _child)
                    try:
                        _child.history_state = self._flatten_history_object_store(session_id)
                    except Exception as exc:
                        # self.logger.warning(f"{_child} is the head node of the session.")
                        self._error_callback(session_id, exc)

                # TODO: make sure that input name in all nodes are different
                _inputs = {
                    ".".join(k.split(".")[1:]): v for k, v in inputs.items() \
                        if ".".join(k.split(".")[1:]) in _child.func_schema.internal.input.all
                }
                _inputs = _child.precheck(session_id=session_id, inputs=_inputs)
                if _inputs:
                    _iterator_target = _child.event_on_execution(
                        session_id=session_id,
                        inputs=_inputs,
                    )
                    _task_dict.update({
                        _iterator_target: {
                            "node_id": _child._id,
                            "node_name": _child.name,
                            "iterator": _iterator_target,
                        }
                    })

            return _task_dict
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

    def _collect_children_tasks(
        self,
        session_id: str,
        output_values_internal: Dict,
        parent: BaseConnectable,
        ):
        try:
            assert self.is_compiled, f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."
            assert any(x is not None for x in output_values_internal.values()), \
                f"{parent.name}: No parameter is filled; all are None."
        except AssertionError as exc:
            self._error_callback(session_id, CancelledError(exc))

        try:
            # EMIT TO CHILDREN of the node
            iterator_dt = dict()
            
            for child_id, edge_info_dict in self.graph.succ[parent._id].items():
                # NODE and EDGE: get child and its mapping keys
                child, _input_keys, _output_keys = self.graph.nodes[child_id]["_obj"], edge_info_dict["_input_keys"], edge_info_dict["_output_keys"]
                # emit value to children: change the name from output into input
                if child.input_state.get(session_id, None) is None:
                    child.input_state[session_id] = {
                        _key: {} for _key in child.func_schema.internal.input.all  # TODO: find a more effient way to do this!
                    }

                _input_values_internal = {
                    s: output_values_internal[d] for s, d in zip(_input_keys, _output_keys) if output_values_internal.get(d, None)  # TODO: 0926
                }
                if _input_values_internal != {}:
                    for _key, _value in _input_values_internal.items():
                        child.input_state[session_id][_key] = child.input_state[session_id].get(_key, {})
                        child.input_state[session_id][_key][parent._id] = {
                            "_timestamp": time.time(),
                            "value": _value,
                        }
                    
                    _inputs = child.precheck(session_id=session_id)
                    if _inputs:
                        self.logger.debug(f"{parent.name} -> emitted to -> {child.name}")
                        iterator_target = child.event_on_execution(
                            session_id=session_id,
                            inputs=_inputs,
                        )
                    
                        iterator_dt.update({
                            iterator_target: {
                                "node_id": child._id, 
                                "node_name": child.name,
                                "iterator": iterator_target,
                                # "task"
                            }
                        })
                    else:
                        self.logger.debug(f"{parent.name} -> emitted to -> {child.name} but NOT executed")
                else:
                    self.logger.debug(f"{parent.name} -> NOT emitted to -> {child.name}: Input emprty")
            return iterator_dt

        except CancelledError as exc:
            self._error_callback(session_id, Exception(f"{parent.name}: Task async_emit has been cancelled. {exc}"))
        except (Exception, BaseException) as exc:
            self._error_callback(session_id, exc)

    def _callback(self, session_id):
        # after execution, self input object store should be cleaned
        self.input_state.pop(session_id, None)  # TODO LC: double check

        # collect from its included components
        _output_values = {}
        for _node_id in self.graph.nodes:
            node = self.graph.nodes[_node_id]["_obj"]
            _output_values[node.name] = node.output_state.get(session_id, None)

        self.output_state[session_id] = self.output_state.get(session_id, [])
        self.output_state[session_id].append({
            "_timestamp": time.time(),
            "_type": self.connectable_type,
            "value": _output_values
        })

        return _output_values

    async def event_on_execution(
        self,
        inputs: Optional[Dict],
        session_id: str,
        **kwargs
    ) -> AsyncGenerator:
        """EXECUTION when the node is triggered."""
        try:
            assert self.is_compiled, f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."
            if inputs:
                # TODO cqju: uncomment to unlock opentelemetry
                if OTEL_ENABLED:
                    span = opentelemetry_tracer._tracer.start_span(self.name)
                    span.set_attributes({
                        "input.value": json.dumps(inputs),  
                    })
                    # TODO: cqju remove for opentelemetry
                    span.set_attributes({
                        "openinference.span.kind": "CHAIN",  
                    })
                    # Creates a Context object with parent set as current span
                    ctx = trace.set_span_in_context(span)
                    # Set as the implicit current context
                    token = context.attach(ctx)

                root_nodes = [self.graph.nodes[_node_id]["_obj"] for _node_id in self.graph.head_nodes]
                task_dict = self._collect_head_tasks(session_id, inputs, root_nodes)
                # compile upon running
                iterator_dict = {
                    asset["iterator"]: {
                        "node_id": asset["node_id"],
                        "node_name": asset["node_name"],
                        "iterator": asset["iterator"], 
                        "task": make_as_task(asset["iterator"]),
                    } for _, asset in task_dict.items()
                }

                while iterator_dict:
                    done_tasks, pending = await wait(
                        [ele["task"] for ele in iterator_dict.values()], return_when=FIRST_COMPLETED
                    )
                    for done_task in done_tasks:
                        node_name, node_id, iterator = next((t["node_name"], t["node_id"], t["iterator"]) for it, t in iterator_dict.items() if t["task"] == done_task)
                        
                        try:
                            response = done_task.result()
                            if response:
                                # IMPORTANT: save context before yield!
                                current_ctx = context.get_current()
                                _output_values: dict = response["value"]
                                _parent = response["node"]
                                # This yield to the final output
                                yield {
                                    "_timestamp": time.time(),
                                    "value": _output_values,
                                    "node": _parent,
                                }
                                # TODO cqju: uncomment to unlock opentelemetry
                                if OTEL_ENABLED:
                                    span.set_attributes({
                                        f'component_output.value.{response["node"].name}': json.dumps(_output_values),  
                                    })
                                context.attach(current_ctx)
                            del response

                        except StopAsyncIteration: # 
                            self.logger.debug(":StopAsyncIteration:")
                            del iterator_dict[iterator]

                            _parent = self.graph.nodes[node_id]["_obj"]
                            # FIXME 1009
                            if isinstance(_parent.output_state[session_id], List):
                                _most_recent_output_values = _parent.output_state[session_id][-1]["value"]  #   1009
                            elif isinstance(_parent.output_state[session_id], Dict):
                                _most_recent_output_values = _parent.output_state[session_id]["value"]  #   1009
                            # TODO 1016
                            if _parent.connectable_type == "Pipeline":
                                _most_recent_output_values = decompose_pipeline(_most_recent_output_values)
                            else:
                                pass

                            if not _parent.is_end:
                                self.logger.warning(f"{_parent.name} -> EMIT")
                                # collect the infomation for children nodes
                                _c_tasks_dict = self._collect_children_tasks(
                                    session_id=session_id,
                                    output_values_internal=_most_recent_output_values,
                                    parent=_parent,
                                )
                                for _, _asset in _c_tasks_dict.items():
                                    iterator_dict.update({
                                        _asset["iterator"]: {
                                            "node_id": _asset["node_id"],
                                            "node_name": _asset["node_name"],
                                            "iterator": _asset["iterator"],
                                            "task": make_as_task(_asset["iterator"])
                                        }
                                    })
                        except Exception as exc:
                            self._error_callback(session_id, exc)
                        else:
                            # The iterator hasn't exhausted or errored out.
                            # Queue the next inspection.
                            iterator_dict[iterator] = {
                                "node_id": node_id,
                                "node_name": node_name,
                                "iterator": iterator,
                                "task": make_as_task(iterator),
                            }
                # Finally, callback function here
                _output_values = self._callback(session_id)
                
                # TODO cqju: uncomment to unlock opentelemetry
                if OTEL_ENABLED:
                    span.set_attributes({
                        "output.value": json.dumps(_output_values),  
                    })
                    span.end()

                # Don't forget to detach or parent will remain the parent above this call stack
                # FIXME cqju
                # context.detach(token)  # ERROR ContextVar

        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
        finally:
            self.count_visited += 1  # TODO cqju