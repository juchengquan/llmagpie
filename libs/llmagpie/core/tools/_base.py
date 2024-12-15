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
from llmagpie.core.utilities.marshal_terable import marshal_iterable, async_marshal_iterable
from llmagpie.core.utilities.marshal_terable import post_run
from llmagpie.core.utilities.async_to_sync import (
    exec_in_event_loop, exec_in_separated_thread
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
    async_function:  Optional[Callable] = None
    """The async function that will be executed when the tool is called."""
    function_type: Literal["sync", "sync_gen", "async", "async_gen"]
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
_ToolResultType = Union[Tuple, Dict, Generator, AsyncGenerator, Exception, BaseException, None]

class Tool(BaseTool):
    def _post_run(self, res: _ToolResultType):  # TODO
        output_model = self.return_schema

        def _async_marshal_iterable(async_res_iterable: AsyncGenerator) -> Generator:
            while True:
                try:
                    yield loop.run_until_complete(async_res_iterable.__anext__())
                except StopAsyncIteration:
                    break
        print(type(res))
        if isinstance(res, Union[Exception, BaseException]):
            raise res
        elif isinstance(res, Generator):
            return marshal_iterable(res, output_model)
        elif isinstance(res, AsyncGenerator):
            loop = new_event_loop()  # TODO
            return marshal_iterable( exec_in_separated_thread(res, loop), output_model)  # TODO
        elif isinstance(res, Dict):
            return output_model(**res if res else {}).model_dump(exclude_none=True)
        elif isinstance(res, Tuple):
            return output_model(**{k:v for k, v in zip(output_model.model_fields.keys(), res)} if res else {}).model_dump(exclude_none=True)
        try:
            return output_model(**{k:v for k, v in zip(output_model.model_fields.keys(), [res])} if res else {}).model_dump(exclude_none=True)
        except:
            raise TypeError("Result type is wrong.")

    def deprecated_run(self, input: dict):  # TODO
        # _input: dict = self.args_schema(**input).model_dump()
        # print("XXX", _input)
        if self.function_type == "sync":
            print("GGG")
            res = self.function(**input)
        else:
            res = self.function(**input)

        print("XXX", res, type(res))

        # res = run_async_as_sync( self.function, **_input)
        # return self._post_run(res)

    async def async_stream(self, input: dict):
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

    async def async_run(self, input: dict):
        res = self.function(**input)
        if isinstance(res, Awaitable):
            res = await res
        
        last_res = None
        if isinstance(res, Generator):
            for e in res:
                last_res = e
        elif isinstance(res, AsyncGenerator):
            async for e in res:
                last_res = e
        elif isinstance(res, Dict):
            last_res = res
        else:
            raise TypeError("WTF")
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

            print("EEE", run_func, iscoroutinefunction(run_func), isasyncgenfunction(run_func))

            input_model = create_schema_from_function(run_func)
            output_model = create_schema_from_types(run_func.__name__, types)

            def _func_wrapper(run_func):
                @wraps(run_func)
                async def _wrapper(*args, **kwargs) -> Union[Dict, Generator, AsyncGenerator]:
                    # TODO 0926
                    inputs = input_model(**kwargs)
                    # res = run_func(inputs.model_dump())
                    res = run_func(*args, **inputs.__dict__)
                    if isinstance(res, Awaitable):
                        res = await res
                    return post_run(res, output_model)

                return _wrapper

            return Tool(
                name=str(_name) if _name else run_func.__name__,
                description=run_func.__doc__,
                function=_func_wrapper(run_func),  # TODO
                # async_function=_func_wrapper(run_func),
                # function_type="async" if iscoroutinefunction(run_func) or isasyncgenfunction(run_func) else "sync",
                function_type="async",
                args_schema=input_model,
                return_schema=output_model,
            )

        return _make_tool

    if run_func:
        return _make_with_name(name)(run_func)
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
            self.result = asyncio_run(self.func(*self.args, **self.kwargs))
        except (Exception, BaseException) as exc:
            self.result = exc

def run_async_as_sync(func, *args, **kwargs):
    try:
        loop = get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        thread = RunThread(func, args, kwargs)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio_run(func(*args, **kwargs))