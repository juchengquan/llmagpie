from __future__ import annotations

import time
from asyncio import CancelledError
from asyncio import get_running_loop, new_event_loop

from pydantic import BaseModel, Field, model_validator

from llmagpie.base.enum import NodeRunningStatus, ConnectableType
from llmagpie.base.connectable import BaseConnectable, FunctionSchema
from llmagpie.base.utils.state import StateResponse
from llmagpie.base.utils.async_to_sync import (
    exec_generator_in_event_loop, exec_generator_in_separated_thread
)
from llmagpie.core.opentelemetry import opentelemetry_tracer
# typing
from typing import (
    cast, final, ClassVar,
    Union, Callable, Awaitable, Dict, Callable, Generator, AsyncGenerator
)


class BaseNode(BaseConnectable):
    """
    A base class for all nodes in the system. It provides basic functionality for
    connecting, disconnecting and managing the state of the node.
    """
    class Config:
        extra = "forbid"
        
    class _ArgsSchemaPlaceholder(BaseModel):
        pass

    connectable_type: ConnectableType = ConnectableType.BASENODE
    is_binded: bool = False
    
    async_call_: ClassVar[Callable]
    input_model_schema: ClassVar[BaseModel] = Field(default_factory=_ArgsSchemaPlaceholder)
    """The schema for the arguments that the tool accepts."""
    output_model_schema: ClassVar[BaseModel]
    """The schema for the arguments that the tool returns."""
    
    def _generate_description_openai(self):
        tool_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model_schema.model_json_schema()
            }
        }
        return tool_schema
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    @final
    async def _async_stream(self, **inputs):
        res = self.async_call_(**inputs)
        if isinstance(res, Awaitable):
            res = await res
        
        if isinstance(res, Generator):
            for e in res:
                yield e
        elif isinstance(res, AsyncGenerator):
            async for e in res:
                yield e
        elif isinstance(res, Dict):
            yield res
        else:
            raise TypeError(
                f"{self.__class__.__name__}.async_call_ must return a dict, Generator, "
                f"AsyncGenerator, or an Awaitable of one of those; got {type(res).__name__}."
            )
    
    @final
    def run(self, **inputs):
        """Runs the node with the provided inputs and yields output.
        """
        res = self.stream(**inputs)
        last_res = None
        for e in res:
            last_res = e
        return last_res
    
    @final
    def stream(self, **inputs):
        """Run the tool with the given inputs and yield the results.
        """
        async_result = cast(AsyncGenerator, self._async_stream(**inputs))
        try:
            _thread_mode = False
            _is_new_loop = False
            aioloop = get_running_loop() # if there is not, go to RuntimeError
            if aioloop and aioloop.is_running():
                _thread_mode = True
                raise RuntimeError
        except RuntimeError:
            _is_new_loop = True
            aioloop = new_event_loop()
        
        if _thread_mode:
            yield from exec_generator_in_separated_thread(async_generator=async_result, loop=aioloop)
        else:
            yield from exec_generator_in_event_loop(async_generator=async_result, loop=aioloop)

        if _is_new_loop:
            aioloop.close()

    @final
    async def async_stream(self, **inputs):
        return self._async_stream(**inputs)

    @final
    async def async_run(self, **inputs):
        res = self._async_stream(**inputs)
        last_res = None
        async for e in res:
            last_res = e
        return last_res    
    
    def _validate(self):
        """
        Validates the binding status of the node and ensures that all required inputs are bound.

        This method performs the following checks:
        1. Ensures that the node has not already been bound to another pipeline.
        2. Marks the node as bound.
        3. If the node is not a start node, it checks that all required input parameters are bound 
           and that no unknown keys are bound.

        Raises:
            AssertionError: If the node has already been bound to another pipeline.
            AssertionError: If the required input parameters are not fully bound or if unknown keys are bound.
        """
        assert not self.is_binded, f"The node has been binded to another pipeline: {self.pipeline}"
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
                    "required": self.input_model_schema.model_json_schema()["required"],
                    "all": self.input_model_schema.model_json_schema()["properties"],
                },
                "output": {
                    "required": [],
                    "all": self.output_model_schema.model_json_schema()["properties"],
                },
            }
        })
        return self

    @opentelemetry_tracer
    async def _async_execute(self, **inputs):
        # parameter checking
        if set(self.func_schema.internal.input.all) != set(inputs.keys()):
            self.logger.warning(
                f'{self.__class__.__name__}:{self.name}: '
                f'Input parameters {set(inputs.keys())} does not align with the keys: '
                f'{set(self.func_schema.internal.input.all)} -> checking required parameters')

            assert set(self.func_schema.internal.input.required).issubset(set(inputs.keys())), \
                (
                    f'{self.__class__.__name__}:{self.name}: Required input parameters missing. '
                    f'{set(self.func_schema.internal.input.required)} does not align with the '
                    f'input keys: {set(inputs.keys())}'
                )

        try:
            return await self.async_call_(**inputs)
        except Exception as exc:
            self.logger.error(f"Error: {exc}")
            raise

    def _callback(self, session_id, _output_values):
        self.iteration_counter[session_id] = self.iteration_counter.get(session_id, 0)
        if self.iteration_counter[session_id] >= self.max_iteration_limit:
            raise Exception("max_iteration_limit is reached.")
        self.iteration_counter[session_id] += 1
            
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

    async def async_event_on_execution(
        self,
        inputs: Dict,
        session_id: str,
        **kwargs
    ) -> AsyncGenerator:
        try:
            self.logger.debug(f"EXECUTE -> {self.name}")
            self._running_status = NodeRunningStatus.RUNNING
            _output_values: Union[Dict, Generator, AsyncGenerator] = await self._async_execute(**inputs)
        except CancelledError as exc:
            exc = Exception(f"{self.name}: The task has been cancelled: {exc}")
            self._error_callback(session_id, exc)
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

        try:
            if isinstance(_output_values, Generator):
                for _v in _output_values:
                    yield StateResponse(
                        timestamp=time.time(),
                        type=self.connectable_type,
                        value=_v,
                        node=self,
                    )
                # only keep the last one
                self._callback(session_id, _v)
            elif isinstance(_output_values, AsyncGenerator):
                async for _v in _output_values:
                    yield StateResponse(
                        timestamp=time.time(),
                        type=self.connectable_type,
                        value=_v,
                        node=self,
                    )
                # only keep the last one
                self._callback(session_id, _v)
            elif isinstance(_output_values, Dict):
                yield StateResponse(
                    timestamp=time.time(),
                    type=self.connectable_type,
                    value=_output_values,
                    node=self,
                )
                self._callback(session_id, _output_values)
            else:
                raise TypeError("Type of output values is wrong.")
            
            self.logger.debug(f'{self.__class__.__name__}:{self.name}: [END] Yielding')
            self._running_status = NodeRunningStatus.INACTIVE
        
        except GeneratorExit as exc:
            self.logger.debug(":GeneratorExit:")
            # pass
            raise exc
        except (Exception, BaseException) as exc:
            self.logger.debug(":BaseException:")
            self._error_callback(session_id, exc)
        finally:
            self.count_visited += 1
