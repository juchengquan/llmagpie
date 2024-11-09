import asyncio
import uuid
import time
from inspect import getfullargspec, signature, _empty
from typing import Callable, Any, cast, Type, Dict, Union
from pydantic import BaseModel, Field, create_model
from pydantic._internal._model_construction import ModelMetaclass
from functools import wraps
from asyncio import create_task


def create_schema_from_function(
    function: Callable,
    name: str = None,
    in_class: bool = False
) -> BaseModel:
    """Create schema from function.

    Args:
        function (Callable): Function
        name (str, optional): Function name. Defaults to None.
        in_class (bool, optional): If the function is in class. Defaults to False.

    Raises:
        ValueError

    Returns:
        BaseModel
    """
    if not name:
        name = function.__name__ + "_Input"
    args = getfullargspec(function)
    # TODO
    if args.varargs or args.varkw:
        raise ValueError("arg of kwargs are not allowed in function definition.")

    parameters = signature(function).parameters
    fields: Dict = {}

    for (idx, p_name), p_val in zip(enumerate(parameters.keys()), parameters.values()):
        if in_class and idx == 0:
            continue
        p_type = p_val.annotation
        p_default = p_val.default

        if p_type is _empty:
            p_type = Any

        if p_default is _empty:
            # Required field
            p_default = Field(...)
        else:
            # Field with pydantic.Field as default value
            p_default = Field(default=cast(p_type, p_default))

        fields[p_name] = (p_type, p_default)

    return create_model(name, **fields)

