from asyncio import get_event_loop
from functools import wraps, partial
from llmagpie.core.function import create_schema_from_function, create_schema_from_types
from llmagpie.core.utilities.marshal_terable import post_run
# typing
from pydantic import BaseModel, create_model, Field
from typing import Union, Optional, Callable, Dict, Awaitable, Callable, Generator, AsyncGenerator
from dataclasses import dataclass
from deprecated import deprecated


def socket_types(run_func: Optional[Callable] = None, **types):
    # class _SchemaConfig:
    #     extra: str = "forbid"
    #     arbitrary_types_allowed: bool = True
    def _func_wrapper(run_func) -> Callable:
        """
        Decorator that sets the output types of the decorated method.

        This happens at class creation time, and since we don't have the decorated
        class available here, we temporarily store the output types as an attribute of
        the decorated method. The ComponentMeta metaclass will use this data to create
        sockets at instance creation time.
        """
        # TODO
        # method_name = run_func.__name__
        # if method_name not in ("run", "run_async"):
        #     raise Exception("'socket_types' decorator can only be used on 'run' and 'run_async' methods")
        input_model = create_schema_from_function(run_func, in_class=True)   # TODO: in_class
        output_model = create_schema_from_types(run_func.__name__, types)

        # TODO async function
        @wraps(run_func)
        async def _wrapper(*args, **kwargs) -> Union[Dict, Generator, AsyncGenerator]:
            # TODO 0926
            inputs = input_model(**kwargs)  # type: ignore
            # res = run_func(inputs.model_dump())
            res = run_func(*args, **inputs.__dict__)  # TODO 1114
            # res = run_func(*args, **inputs.model_dump())  # Note: need to take args as it includes `self`
            if isinstance(res, Awaitable):
                res = await res

            return post_run(res, output_model)
        # set up searchable object
        setattr(_wrapper, "_input_model", input_model)
        setattr(_wrapper, "_output_model", output_model)
        return _wrapper

    if run_func:
        # Decorator is called without parens
        return _func_wrapper(run_func)
    return _func_wrapper



@deprecated
def conditional(run_func: Optional[Callable] = None, **types):
    # return type of conditional function must be boolean.
    def _decorator(run_func):
        # TODO
        # method_name = run_func.__name__
        # if method_name not in ("run", "run_async"):
        #     raise Exception("'socket_types' decorator can only be used on 'run' and 'run_async' methods")

        input_model = create_schema_from_function(run_func, in_class=False)  # TODO: in_class

        # TODO async function
        @wraps(run_func)
        def wrapper(*args, **kwargs):
            # TODO 0926
            inputs = input_model(**kwargs)  # type: ignore
            res = run_func(inputs.model_dump())
            # if isinstance(res, Awaitable):
            #     res = await res
            assert isinstance(res, bool), "Output must be boolean!"
            return res

        # set up searchable object
        setattr(wrapper, "_input_model", input_model)
        return wrapper

    if run_func:
        # Decorator is called without parens
        return _decorator(run_func)

    return _decorator

@deprecated(reason="Not used")
def convert_as_func_w_internal_variables(func: Optional[Callable] = None, mapping: Dict = {}):
    """Convert function with internal variables.
    """
    if func is None:
        return partial(convert_as_func_w_internal_variables, mapping=mapping)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper
        """
        _kwargs = {mapping[k]: v for k, v in kwargs.items()}
        return func(*args, **_kwargs)

    return wrapper
