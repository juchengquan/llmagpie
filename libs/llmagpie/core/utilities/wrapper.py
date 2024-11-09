from functools import wraps, partial
from llmagpie.core.function import create_schema_from_function
# typing
from pydantic import BaseModel, create_model, Field
from typing import Optional, Callable, Dict, Awaitable, Callable
from dataclasses import dataclass

from deprecated import deprecated

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
            inputs = input_model(**kwargs)
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


def socket_types(run_func: Optional[Callable] = None, **types):
    def socket_types_decorator(run_func):
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

        _schema = {_n: (_t, Field(default=None, required=True)) for _n, _t in types.items()}
        output_model = create_model(run_func.__name__ + "_Output", **_schema)

        # TODO async function
        @wraps(run_func)
        async def wrapper(*args, **kwargs):
            # TODO 0926
            inputs = input_model(**kwargs)
            res = run_func(*args, **inputs.model_dump())  # Note: need to take args as it includes `self`
            if isinstance(res, Awaitable):
                res = await res
                if isinstance(res, tuple):
                    res = {k: v for d in res for k, v in d.items()}
            return output_model(**res if res else {})  # .model_dump()  # TODO: 0926: exclude_none=True

        # set up searchable object
        setattr(wrapper, "_input_model", input_model)
        setattr(wrapper, "_output_model", output_model)
        return wrapper

    if run_func:
        # Decorator is called without parens
        return socket_types_decorator(run_func)

    return socket_types_decorator


@deprecated(reason="Not used")
def convert_as_func_w_internal_variables(func: Callable = None, mapping: Dict = None):
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
