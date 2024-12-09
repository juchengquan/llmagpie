from __future__ import annotations
# import os
import uuid
import time

from inspect import isawaitable
from abc import abstractmethod
from asyncio import CancelledError

from pydantic import BaseModel, Field, PrivateAttr, model_validator, computed_field
from pydantic._internal._model_construction import ModelMetaclass

from llmagpie.core.connectable import BaseConnectable, FunctionSchema
from llmagpie.core.connectable._base import _RunningStatus
# EXPERIMENTAL
from llmagpie.experimental.opentelemetry import opentelemetry_tracer
# typing
from typing import List, Dict, Union, Awaitable, Set, Optional, Callable, Generator, AsyncGenerator, Coroutine


class BaseNode(BaseConnectable):
    class Config:
        extra = "forbid"

    connectable_type: str = "BaseNode"
    is_binded: bool = False

    def _validate(self):  # TODO: may need to change function name
        assert not self.is_binded, "The node has been binded to another pipeline."
        self.is_binded = True
        
        # Check input bound status
        if not self.is_start:
            assert set(self.func_schema.internal.input.required).issubset(set(self._input_keys_binded)) \
                and set(self._input_keys_binded).issubset(set(self.func_schema.internal.input.all)), \
                "Required inputs parameters are not fully bound. Or unknown keys bound."
    
    @model_validator(mode="after")
    def _contruct_schemas(self):
        self.func_schema = FunctionSchema(**{
            "internal": {
                "input": {
                    "required": self.async_call._input_model.schema()["required"],
                    "all": self.async_call._input_model.schema()["properties"],
                },
                "output": {
                    "required": [],
                    "all": self.async_call._output_model.schema()["properties"],
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
        return self
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # CHILDREN PROCESS
    @abstractmethod
    async def async_call(self):
        """async call"""
        raise NotImplementedError

    @opentelemetry_tracer
    async def _execute(self, **kwargs):
        try:
            # parameter checking
            if set(self.func_schema.internal.input.all) != set(kwargs.keys()):
                assert set(self.func_schema.internal.input.required).issubset(set(kwargs.keys())), \
                    f'{self.__class__.__name__}:{self.name}: Required input parameters missing. {set(self.func_schema.internal.input.required)} does not align with the input keys: {set(kwargs.keys())}'

                self.logger.warning(f'{self.__class__.__name__}:{self.name}: Input pamatemeters {set(kwargs.keys())} does not align with the keys: {set(self.func_schema.internal.input.all)} \
                    -> checking required parameters')
        except AssertionError as exc:
            raise exc

        try:
            _output_values = self.async_call(**kwargs) # type: ignore
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
        self.output_state[session_id] = self.output_state.get(session_id, [])
        self.output_state[session_id].append({
            "_timestamp": time.time(),
            "_type": self.connectable_type,
            "value": _output_values
        })
        
        self.output_history_state[session_id] = self.output_history_state.get(session_id, [])
        self.output_history_state[session_id].append({
            "_timestamp": time.time(),
            "_type": self.connectable_type,
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
            self._running_status = _RunningStatus.RUNNING
            # TODO: iteration
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
                        "_type": self.connectable_type,
                        "value": _v,
                        "node": self,
                    }
                # only keep the last one
                self._callback(session_id, _v)
            else: # isinstance(_output_values, Dict)
                yield {
                    "_timestamp": time.time(),
                    "_type": self.connectable_type,
                    "value": _output_values,
                    "node": self,
                }

                self._callback(session_id, _output_values)
            
            self.logger.warning(f'{self.__class__.__name__}:{self.name}: [END] Yielding')
            self._running_status = _RunningStatus.INACTIVE
        
        except GeneratorExit as exc:
            self.logger.debug(":GeneratorExit:")
            # pass
            raise exc
        except (Exception, BaseException) as exc:
            self.logger.debug(":BaseException:")
            self._error_callback(session_id, exc)
        finally:
            self.count_visited += 1  # TODO cqju
