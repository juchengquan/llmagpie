from collections.abc import Callable
from inspect import _empty, getfullargspec, signature
from typing import Annotated, Any, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model


def _strip_titles(schema: dict[str, Any], model) -> None:
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)


# Pydantic >=2.11 dropped support for passing a class as `__config__` to
# `create_model`; it must be a ConfigDict (dict-like) instead.
_SCHEMA_CONFIG: ConfigDict = ConfigDict(
    extra="ignore",
    arbitrary_types_allowed=True,
    json_schema_extra=_strip_titles,
)


def create_schema_from_types(name: str, types: dict) -> type[BaseModel]:
    """Create schema from types.

    Args:
        types (Dict): Types dictionary

    Raises:
        ValueError

    Returns:
        BaseModel
    """
    fields: dict = {}
    for p_name, p_val in types.items():
        # if in_class and idx == 0:
        #     continue
        p_type = p_val
        p_description = None
        if get_origin(p_type) == Annotated:
            p_description = p_type.__metadata__[0]
            p_type = p_type.__origin__

        p_field = Field(default=None, description=p_description)

        fields[p_name] = (p_type, p_field)
    return create_model(name + "_Output", __config__=_SCHEMA_CONFIG, **fields)  # type: ignore


def create_schema_from_function(
    function: Callable,
    in_class: bool = False,
) -> type[BaseModel]:
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
    if args.varargs or args.varkw:
        raise ValueError("arg of kwargs are not allowed in function definition.")
    # assert "return" in args.annotations, "Return value type is not declared."

    parameters = signature(function).parameters

    fields: dict = {}
    for idx, (p_name, p_val) in enumerate(parameters.items()):
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
            p_field = Field(default=p_default, description=p_description)

        fields[p_name] = (p_type, p_field)

    return create_model(function_name, __config__=_SCHEMA_CONFIG, **fields)  # type: ignore
