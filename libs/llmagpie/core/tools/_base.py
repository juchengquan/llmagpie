import asyncio
from asyncio import get_event_loop
from abc import ABC, abstractmethod
from pydantic import BaseModel, create_model, Field
from inspect import getfullargspec, iscoroutinefunction
from typing import Type, Literal, Any, Union, Optional, Callable, Dict, Tuple, Awaitable, Callable, Iterable, Generator, AsyncGenerator, Annotated, get_origin

from llmagpie.core.function import create_schema_from_function


class BaseTool(BaseModel, ABC):
    class Config:
        extra: Any = "forbid"
        arbitrary_types_allowed: bool = True

    class _ArgsSchemaPlaceholder(BaseModel):
        pass

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool."""
    args_schema: Type[BaseModel] = Field(default_factory=_ArgsSchemaPlaceholder)
    """The schema for the arguments that the tool accepts."""
    return_schema: Type[Any]
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

class Tool(BaseTool):
    def _post_run(self, res: Union[Generator, AsyncGenerator, Dict, Tuple, Any]):  # TODO
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
            
class _SchemaConfig:
    extra: Any = "forbid"
    arbitrary_types_allowed: bool = True

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model) -> None:
        for prop in schema.get('properties', {}).values():
            prop.pop('title', None)

def tool(func: Optional[Union[Callable, Awaitable]] = None, name: Optional[str] = None, **types):
    def _make_with_name(_name) -> Callable:
        def _make_tool(func) -> BaseTool:
            args = getfullargspec(func)
            if args.varargs or args.varkw:
                raise ValueError("arg of kwargs are not allowed in function definition.")

            fields: Dict = {}
            for (idx, p_name), p_val in zip(enumerate(types.keys()), types.values()):
                # if in_class and idx == 0:
                #     continue
                p_type = p_val
                p_description = None
                if get_origin(p_type) == Annotated:
                    p_description = p_type.__metadata__[0]
                    p_type = p_type.__origin__

                p_field = Field(default=None, description=p_description)
            
                fields[p_name] = (p_type, p_field)
            # _schema = {_n: (_t, Field(default=None, required=True)) for _n, _t in types.items()}
            return_schema = create_model(func.__name__ + "_Output", **fields, __config__=_SchemaConfig)  # type: ignore

            return Tool(
                name=str(name) if name else func.__name__,
                description=func.__doc__,
                function=func,
                function_type="async" if iscoroutinefunction(func) else "sync",
                args_schema=create_schema_from_function(func),  # type: ignore
                return_schema=return_schema,
            )
        return _make_tool

    if func:
        return _make_with_name(name)(func)
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