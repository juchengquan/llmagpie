import asyncio
import uuid
import time
from inspect import getfullargspec, signature, _empty
from pydantic import BaseModel, Field, create_model
from pydantic._internal._model_construction import ModelMetaclass
from functools import wraps
from asyncio import create_task
from typing import Callable, Any, cast, Type, Dict, Union, Annotated, get_origin


class _SchemaConfig:
    extra: Any = "forbid"
    arbitrary_types_allowed: bool = True

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model) -> None:
        for prop in schema.get('properties', {}).values():
            prop.pop('title', None)


def create_schema_from_function(
    function: Callable,
    in_class: bool = False,
) -> BaseModel:
    """Create schema from function.

    Args:
        function (Callable): Function
        in_class (bool, optional): If the function is in class. Defaults to False.

    Raises:
        ValueError

    Returns:
        BaseModel
    """
    function_name = function.__name__ + "_Input"
    args = getfullargspec(function)
    # assert "return" in args.annotations, "Return value type is not declared."

    if args.varargs or args.varkw:
        raise ValueError("arg of kwargs are not allowed in function definition.")

    parameters = signature(function).parameters
    fields: Dict = {}

    for (idx, p_name), p_val in zip(enumerate(parameters.keys()), parameters.values()):
        if in_class and idx == 0:
            continue
        p_type = p_val.annotation
        p_description = None
        if get_origin(p_type) == Annotated:
            p_description = p_type.__metadata__[0]
            p_type = p_type.__origin__

        p_default = p_val.default

        if p_type is _empty:
            p_type = Any

        if p_default is _empty:
            # Required field
            p_field = Field(description=p_description)
        else:
            # Field with pydantic.Field as default value
            p_field = Field(default=cast(p_type, p_default), description=p_description)
    
        fields[p_name] = (p_type, p_field)

    return create_model(function_name, **fields, __config__=_SchemaConfig)