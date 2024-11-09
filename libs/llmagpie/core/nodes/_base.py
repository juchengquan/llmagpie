from __future__ import annotations
# import os
import uuid
import asyncio
import time

from inspect import isawaitable
from abc import abstractmethod
from asyncio import create_task, CancelledError
from types import MethodType

from pydantic import BaseModel, Field, PrivateAttr, model_validator, computed_field
from pydantic._internal._model_construction import ModelMetaclass

from llmagpie.core.nodes.disposable import BaseNodeDisposable
from llmagpie.core.dag import SingleDAG
from llmagpie.core.connectable import BaseConnectable

# EXPERIMENTAL
from llmagpie.exp.opentelemetry import opentelemetry_tracer
# typing
from typing import List, Dict, Union, Any, Awaitable, Set, Optional, AsyncIterator, Callable

class BaseNode(BaseConnectable):
    node_type: str = "Node"

    cond_func: Union[bool, Callable] = None
    inputs_to_cond: Dict = None

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
        except AssertionError as err:
            raise err

        try:
            _output_values: Union[Awaitable, BaseModel] = self.async_call(**kwargs)  # cqju: overwrite in children
            if isawaitable(_output_values):
                _output_values: BaseModel = await _output_values

            return _output_values.model_dump(exclude_none=True)
        except (Exception, BaseException) as exc:
            self.logger.error(f"Error: {str(exc)}")
            raise exc
        # finally:
            """"""

    def precheck(
        self,
        session_id: str,
        inputs: Dict = None,
        **kwargs
    ) -> Union[Dict, None]:
        try:
            if inputs is None:
                inputs = self._get_local_store(session_id)

            if self.cond_func is not None:
                if self.inputs_to_cond:
                    inputs_to_cond_values = {k: inputs[v] for k, v in self.inputs_to_cond.items()}
                    self.logger.debug("inputs_to_cond_values: ", inputs_to_cond_values)
                    if self.cond_func(**inputs_to_cond_values) == False:
                        self.logger.warning(f"[Condition not met] NOT EXECUTED -> {self.name}")
                        return None
                else:
                    self.logger.warning("Missing inputs_to_cond.")
                    assert type(self.cond_func) == bool
                    if self.cond_func == False:
                        self.logger.warning(f"[Condition not met] NOT EXECUTED -> {self.name}")
                        return None
            
            # Patches
            if isinstance(inputs, Union[ModelMetaclass, BaseModel]):
                inputs = inputs.model_dump()  # exclude_none=True

            if not (
                set(inputs.keys()).issubset(self._input_schema_all["internal"]) and \
                set(self._input_schema_required["internal"]).issubset(set(inputs.keys()))
            ):  
                self.logger.warning(f'{self.__class__.__name__}:{self.name}: Input pamatemeters {set(inputs.keys())} does not align with the keys: {set(self._input_schema_all["internal"])}\
                or Required input parameters {set(self._input_schema_required["internal"])} does not align with the input keys: {set(inputs.keys())}')
                
                self.logger.warning(f"[PRECHECK] NOT EXECUTED YET -> {self.name}")
                return None

            return inputs
        except (BaseException, Exception) as err:
            self.clean_object_store(session_id)
            raise Exception(err)

    async def event_on_execution(
        self,
        inputs: Dict,
        session_id: str,
        **kwargs
    ) -> AsyncIterator:
        """EXECUTION when the node is triggered."""
        try:
            self.logger.debug(f"EXECUTE -> {self.name}")
            # TODO 0926
            self.iteration_counter[session_id] = self.iteration_counter.get(session_id, 0)
            if self.iteration_counter[session_id] >= self.max_iteration_limit:
                raise Exception("max_iteration_limit is reached.")

            _output_values: dict = await self._execute(**inputs)  # BaseModel
            self.iteration_counter[session_id] += 1

        except CancelledError as err:
            self.clean_object_store(session_id)
            raise Exception(f"{self.name}: The task has been cancelled: {err}")
        except (BaseException, Exception) as err:
            self.clean_object_store(session_id)
            raise Exception(err)

        try:
            _timestamp = time.time()

            # after execution, input object store should be cleaned
            self.input_object_store.pop(session_id, None) # TODO LC: double check

            # FIXME 1009
            self.output_object_store[session_id] = self.output_object_store.get(session_id, [])
            self.output_object_store[session_id].append({
                "_timestamp": _timestamp,
                "_type": self.node_type,
                "value": _output_values
            })

            yield {
                "_timestamp": _timestamp,
                "_type": self.node_type,
                "value": _output_values,
                "node": self,
            }
            self.logger.debug(f"[END] Yielding... {self.name}")
        
        except GeneratorExit:
            self.logger.debug(":GeneratorExit:")
            # pass
            raise err
        except (Exception, BaseException) as err:
            self.logger.debug(":BaseException:")
            self.logger.error(err)
            self.clean_object_store(session_id)
            raise err
        finally:
            self.count_visited += 1  # TODO cqju