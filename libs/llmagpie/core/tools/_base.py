import functools
from asyncio import (
    get_running_loop,
    new_event_loop,
    run as asyncio_run
)
from abc import ABC, abstractmethod
from pydantic import BaseModel, create_model, Field
from inspect import getfullargspec, iscoroutinefunction, isasyncgenfunction
from typing import (cast, Any, Type, Literal, Union, Optional, Callable, Awaitable, Dict, Tuple, Callable, Generator, AsyncGenerator)
from functools import wraps

from llmagpie.core.function import create_schema_from_function, create_schema_from_types
from llmagpie.core.utilities.marshal_terable import post_run
from llmagpie.core.utilities.async_to_sync import (
    exec_generator_in_event_loop, exec_generator_in_separated_thread
)

class BaseTool(BaseModel, ABC):
    class Config:
        extra: str = "forbid"
        arbitrary_types_allowed: bool = True

    class _ArgsSchemaPlaceholder(BaseModel):
        pass

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool."""
    args_schema: Type[BaseModel] = Field(default_factory=_ArgsSchemaPlaceholder)
    """The schema for the arguments that the tool accepts."""
    return_schema: Type[BaseModel]
    """The schema for the arguments that the tool returns."""
    function: Any
    """The function that will be executed when the tool is called."""
    # async_function:  Optional[Callable] = None
    # """The async function that will be executed when the tool is called."""
    function_type: Literal["sync", "async"]
    
    def _generate_description_openai(self):
        tool_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.schema()
            }
        }
        return tool_schema

class Tool(BaseTool):
    async def _stream(self, input: dict):
        res = self.function(**input)
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
            raise TypeError("WTF")       
        
    def run(self, input: dict):
        res = self.stream(input)
        last_res = None
        for e in res:
            last_res = e
        return last_res
    
    def stream(self, input: dict):
        async_result = cast(AsyncGenerator, self._stream(input))
        
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
        
        try:
            if _thread_mode:
                yield from exec_generator_in_separated_thread(async_generator=async_result, loop=aioloop)
            else:
                yield from exec_generator_in_event_loop(async_generator=async_result, loop=aioloop)

            if _is_new_loop:
                aioloop.close()
        except Exception as exc:
            raise exc
  
    async def async_stream(self, input: dict):
        return self._stream(input)

    async def async_run(self, input: dict):
        res = self._stream(input)
        last_res = None
        async for e in res:
            last_res = e
        return last_res    

# decorator
def as_tool(run_func: Optional[Callable] = None, name: Optional[str] = None, **types):
    def _make_with_name(_name) -> Callable:
        def _make_tool(run_func: Callable) -> BaseTool:
            args = getfullargspec(run_func)
            if args.varargs or args.varkw:
                raise ValueError("arg of kwargs are not allowed in function definition.")
            if not run_func.__doc__:
                raise ValueError("Tools does not have description.")

            input_model = create_schema_from_function(run_func)
            output_model = create_schema_from_types(run_func.__name__, types)

            def _func_wrapper(run_func):
                @wraps(run_func)
                def _wrapper(*args, **kwargs) -> Union[Dict, Generator, AsyncGenerator]:
                    # TODO 0926
                    inputs = input_model(**kwargs)
                    # res = run_func(inputs.model_dump())
                    res = run_func(*args, **inputs.__dict__)
                    # if isinstance(res, Awaitable):
                    #     res = await res
                    return post_run(res, output_model)
                
                @wraps(run_func)
                async def _async_wrapper(*args, **kwargs) -> Union[Dict, Generator, AsyncGenerator]:
                    # TODO 0926
                    inputs = input_model(**kwargs)
                    # res = run_func(inputs.model_dump())
                    res = run_func(*args, **inputs.__dict__)
                    if isinstance(res, Awaitable):
                        res = await res
                    return post_run(res, output_model)

                if iscoroutinefunction(run_func) or isasyncgenfunction(run_func):
                    return _async_wrapper
                return _wrapper 

            return Tool(
                name=str(_name) if _name else run_func.__name__,
                description=run_func.__doc__,
                function=_func_wrapper(run_func),
                function_type="async" if iscoroutinefunction(run_func) or isasyncgenfunction(run_func) else "sync",
                args_schema=input_model,
                return_schema=output_model,
            )

        return _make_tool

    if run_func:
        return _make_with_name(name)(run_func)
    return _make_with_name(name)
