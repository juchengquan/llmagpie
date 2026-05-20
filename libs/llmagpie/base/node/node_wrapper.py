from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from functools import partial, wraps
from inspect import getfullargspec

# typing
from typing import cast

from pydantic import BaseModel, ConfigDict

from ._base import BaseNode
from ._post_run import post_run
from ._schema import create_schema_from_function, create_schema_from_types


class MakeNode:
    @classmethod
    def from_class(
        cls, class_: type | None = None, func_name: str = "", outputs: dict | None = None
    ):
        outputs = outputs or {}

        def _cls_wrapper(class_, func_name: str, outputs: dict):
            assert func_name not in [
                "run",
                "stream",
                "async_run",
                "async_stream",
                "_stream",
                "async_call_",
            ], "The function name has conflict with the default function name."
            assert func_name and hasattr(class_, func_name), "function name is wrong."
            func_callable = getattr(class_, func_name)
            input_model = create_schema_from_function(func_callable, in_class=True)
            output_model = create_schema_from_types(func_callable.__name__, outputs)

            @wraps(func_callable)
            async def _wrapper(*args, **kwargs):
                inputs = input_model(**kwargs)  # type: ignore
                res = func_callable(*args, **inputs.__dict__)
                if isinstance(res, Awaitable):
                    res = await res
                return post_run(res, output_model)

            # classvar binding
            class_.async_call_ = _wrapper
            class_.input_model_schema = input_model
            class_.output_model_schema = output_model

            return class_

        if class_:
            return _cls_wrapper(class_, func_name, outputs)
        return partial(_cls_wrapper, func_name=func_name, outputs=outputs)

    @classmethod
    def from_function(
        cls, func: Callable | None = None, name: str | None = None, outputs: dict | None = None
    ):
        outputs = outputs or {}

        def _make_with_name(_name) -> Callable:
            def _make_node(func: Callable) -> BaseModel:
                args = getfullargspec(func)
                if args.varargs or args.varkw:
                    raise ValueError("arg of kwargs are not allowed in function definition.")
                if not func.__doc__:
                    raise ValueError("Tools does not have description.")

                input_model = create_schema_from_function(func)
                output_model = create_schema_from_types(func.__name__, outputs)

                def _func_wrapper(func):
                    @wraps(func)
                    async def _async_wrapper(*args, **kwargs) -> dict | Generator | AsyncGenerator:
                        inputs = input_model(**kwargs)
                        res = func(*args, **inputs.__dict__)
                        if isinstance(res, Awaitable):
                            res = await res
                        return post_run(res, output_model)

                    # if iscoroutinefunction(func) or isasyncgenfunction(func):
                    #     return _async_wrapper
                    return _async_wrapper

                class AsNode(BaseNode):
                    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

                    async_call_ = staticmethod(_func_wrapper(func))
                    input_model_schema = cast(BaseModel, input_model)
                    output_model_schema = cast(BaseModel, output_model)

                return AsNode(
                    name=str(_name) if _name else func.__name__,
                    description=func.__doc__,
                    # function_type="async" if iscoroutinefunction(func) or isasyncgenfunction(func) else "sync",
                )

            return _make_node

        if func:
            return _make_with_name(name)(func)
        return _make_with_name(name)


def from_class(class_: type | None = None, func_name: str = "", outputs: dict | None = None):
    return MakeNode.from_class(class_=class_, func_name=func_name, outputs=outputs or {})


def from_function(
    func: Callable | None = None, name: str | None = None, outputs: dict | None = None
):
    return MakeNode.from_function(func=func, name=name, outputs=outputs or {})
