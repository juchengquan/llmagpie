from __future__ import annotations
# import os
import uuid
import time

from inspect import isawaitable
from abc import abstractmethod
from asyncio import CancelledError

from pydantic import BaseModel, Field, PrivateAttr, model_validator, computed_field
from pydantic._internal._model_construction import ModelMetaclass

from llmagpie.core.nodes.disposable import BaseNodeDisposable
from llmagpie.core.dag import SingleDAG
from llmagpie.core.connectable import BaseConnectable

# EXPERIMENTAL
from llmagpie.experimental.opentelemetry import opentelemetry_tracer
# typing
from typing import List, Dict, Union, Any, Awaitable, Set, Optional, Callable, Generator, AsyncGenerator

class BaseNode(BaseConnectable):
    class Config:
        extra = "allow"

    node_type: str = "Node"

    @computed_field
    @property
    def _input_schema_required(self) -> Dict:
        """"""
        _prop: List = self.async_call._input_model.schema()["required"]
        return {
            "internal": _prop,
            "external": [
                f"{self.name}.{k}" for k in _prop
            ]
        }

    @computed_field
    @property
    def _input_schema_all(self) -> Dict:
        """"""
        _prop = self.async_call._input_model.schema()["properties"]
        return {
            "internal": _prop,
            "external": {
                f"{self.name}.{k}": v for k, v in _prop.items()
            }
        }

    @computed_field
    @property
    def _output_schema_all(self) -> Dict:
        """"""
        _prop = self.async_call._output_model.schema()["properties"]
        return {
            "internal": _prop,
            "external": {
                f"{self.name}.{k}": v for k, v in _prop.items()
            }
        }
        
    @computed_field
    @property
    def _output_schema_required(self) -> Dict:
        """"""
        _prop = self.async_call._output_model.schema()["required"]
        return {
            "internal": _prop,
            "external": [
                f"{self.name}.{k}" for k in _prop
            ]
        }

    @model_validator(mode="after")
    def validate_output_keys(self):
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _validate(self):  # TODO
        # Check in-degree and out-degree of nodes (on pipeline)
        assert (self.pipeline.graph.in_degree(self._id) == 0 and self.is_start is True) \
            or (self.pipeline.graph.in_degree(self._id) != 0 and self.is_start is not True), \
            f"{self.__class__.__name__} Node In-degree is wrong."
        assert (self.pipeline.graph.out_degree(self._id) == 0 and self.is_end is True) \
            or (self.pipeline.graph.out_degree(self._id) != 0 and self.is_end is not True), \
            f"{self.__class__.__name__} Node Out-degree is wrong."

        # Check input bound status
        if not self.is_start:
            assert set(self._input_schema_required["internal"]).issubset(set(self._input_keys_binded)) \
                and set(self._input_keys_binded).issubset(set(self._input_schema_all["internal"])), \
                "Required inputs parameters are not fully bound. Or unknown keys bound."
    
    # CHILDREN PROCESS
    @abstractmethod
    async def async_call(self):
        """async call"""
        raise NotImplementedError

    @opentelemetry_tracer
    async def _execute(self, **kwargs):
        try:
            # parameter checking
            if set(self._input_schema_all["internal"]) != set(kwargs.keys()):
                assert set(self._input_schema_required["internal"]).issubset(set(kwargs.keys())), \
                    f'{self.__class__.__name__}:{self.name}: Required input parameters missing. {set(self._input_schema_required["internal"])} does not align with the input keys: {set(kwargs.keys())}'

                self.logger.warning(f'{self.__class__.__name__}:{self.name}: Input pamatemeters {set(kwargs.keys())} does not align with the keys: {set(self._input_schema_all["internal"])}\
                    -> checking required parameters')
        except AssertionError as exc:
            raise exc

        try:
            _output_values: Union[Awaitable, BaseModel, Generator] = self.async_call(**kwargs)  # cqju: overwrite in children
            if isawaitable(_output_values):
                _output_values: dict = await _output_values

            return _output_values
        except (Exception, BaseException) as exc:
            self.logger.error(f"Error: {str(exc)}")
            raise exc
        # finally:
            """"""

    def _callback(self, session_id, _output_values):
        # after execution, self input object store should be cleaned
        self.input_object_store.pop(session_id, None)  # TODO LC: double check

        # collect from its included components
        # _output_values = {}
        # for _node_id in self.graph.nodes:
        #     node = self.graph.nodes[_node_id]["_obj"]
        #     _output_values[node.name] = node.output_object_store.get(session_id, None)

        self.output_object_store[session_id] = self.output_object_store.get(session_id, [])
        self.output_object_store[session_id].append({
            "_timestamp": time.time(),
            "_type": self.node_type,
            "value": _output_values
        })

        return _output_values

    async def event_on_execution(
        self,
        inputs: Dict,
        session_id: str,
        **kwargs
    ) -> AsyncGenerator:
        """EXECUTION when the node is triggered."""
        try:
            self.logger.debug(f"EXECUTE -> {self.name}")
            # TODO 0926
            self.iteration_counter[session_id] = self.iteration_counter.get(session_id, 0)
            if self.iteration_counter[session_id] >= self.max_iteration_limit:
                raise Exception("max_iteration_limit is reached.")

            _output_values: Union[Dict, Generator[Dict]] = await self._execute(**inputs)
            self.iteration_counter[session_id] += 1

        except CancelledError as exc:
            exc = Exception(f"{self.name}: The task has been cancelled: {exc}")
            self._error_callback(session_id, exc)
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

        try:
            # TODO
            if isinstance(_output_values, Generator):
                for _v in _output_values:
                    yield {
                        "_timestamp": time.time(),
                        "_type": self.node_type,
                        "value": _v,
                        "node": self,
                    }
                # only keep the last one
                self._callback(session_id, _v)
            else: # isinstance(_output_values, Dict)
                yield {
                    "_timestamp": time.time(),
                    "_type": self.node_type,
                    "value": _output_values,
                    "node": self,
                }

                self._callback(session_id, _output_values)
            
            self.logger.warning(f"[END] Yielding... {self.name}")
        
        except GeneratorExit as exc:
            self.logger.debug(":GeneratorExit:")
            # pass
            raise exc
        except (Exception, BaseException) as exc:
            self.logger.debug(":BaseException:")
            self._error_callback(session_id, exc)
        finally:
            self.count_visited += 1  # TODO cqju
