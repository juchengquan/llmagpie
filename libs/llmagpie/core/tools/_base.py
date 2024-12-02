import asyncio
from asyncio import get_event_loop
from abc import ABC, abstractmethod
from pydantic import BaseModel, create_model, Field
from inspect import getfullargspec, iscoroutinefunction
from typing import Type, Literal, Union, Optional, Callable, Dict, Tuple, Awaitable, Callable, Iterable, Generator, AsyncGenerator

from llmagpie.core.function import create_schema_from_function, create_schema_from_types


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
    function: Callable
    """The function that will be executed when the tool is called."""
    function_type: Literal["sync", "async"]
    # async_function: Optional[Awaitable] = None
    """The async function that will be executed when the tool is called."""

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

# TODO
_ToolResultType = Union[Generator, AsyncGenerator, Dict, Tuple, Exception, BaseException, None]

class Tool(BaseTool):
    def _post_run(self, res: _ToolResultType):  # TODO
        output_model = self.return_schema
        
        def _marshal_iterable(res_iterable: Generator) -> Generator:
            for _res in res_iterable:
                yield output_model(**_res if _res else {}).model_dump(exclude_none=True)  # TODO: 0926: exclude_none=True

        def _async_to_sync_marshal_iterable(async_res_iterable: AsyncGenerator) -> Generator:  # nest_asyncio
            loop = get_event_loop()
            while True:
                try:
                    yield loop.run_until_complete(async_res_iterable.__anext__())
                except StopAsyncIteration:
                    break

        if isinstance(res, Union[Exception, BaseException]):
            raise res
            
        if isinstance(res, Generator):
            return _marshal_iterable(res)
        if isinstance(res, AsyncGenerator):
            return _async_to_sync_marshal_iterable(res)
        if isinstance(res, Dict):
            return output_model(**res if res else {}).model_dump(exclude_none=True)
        if isinstance(res, Tuple):
            return output_model(**{k:v for k, v in zip(output_model.model_fields.keys(), res)} if res else {}).model_dump(exclude_none=True)
        
        try:
            return output_model(**{k:v for k, v in zip(output_model.model_fields.keys(), [res])} if res else {}).model_dump(exclude_none=True)
        except:
            raise TypeError("Result type is wrong.")
        
    def run(self, input: dict):
        _input: dict = self.args_schema(**input).model_dump()
        
        if self.function_type == "sync":
            res = self.function(**_input)
        else:
            res = run_async_as_sync( self.function, **_input)
            
        return self._post_run(res)


    async def async_run(self, input: dict):
        _input: dict = self.args_schema(**input).model_dump()
        
        if self.function_type == "sync":
            res = self.function(**_input)
        else:
            res = await self.function(**_input)
        
        return self._post_run(res)

def tool(fun_func: Optional[Union[Callable, Awaitable]] = None, name: Optional[str] = None, **types):
    def _make_with_name(_name) -> Callable:
        def _make_tool(fun_func) -> BaseTool:
            args = getfullargspec(fun_func)
            if args.varargs or args.varkw:
                raise ValueError("arg of kwargs are not allowed in function definition.")

            return Tool(
                name=str(_name) if _name else fun_func.__name__,
                description=fun_func.__doc__,
                function=fun_func,
                function_type="async" if iscoroutinefunction(fun_func) else "sync",
                args_schema=create_schema_from_function(fun_func),
                return_schema=create_schema_from_types(fun_func.__name__, types),
            )
        return _make_tool

    if fun_func:
        return _make_with_name(name)(fun_func)
    else:
        return _make_with_name(name)


import threading

class RunThread(threading.Thread):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        super().__init__()

    def run(self):
        try:
            self.result = asyncio.run(self.func(*self.args, **self.kwargs))
        except (Exception, BaseException) as exc:
            self.result = exc

def run_async_as_sync(func, *args, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        thread = RunThread(func, args, kwargs)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func(*args, **kwargs))