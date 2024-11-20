from asyncio import get_event_loop
from functools import wraps, partial
from llmagpie.core.function import create_schema_from_function
# typing
from pydantic import BaseModel, create_model, Field
from typing import Union, Optional, Callable, Dict, Tuple, Awaitable, Callable, Iterable, Generator, AsyncGenerator
from dataclasses import dataclass
from deprecated import deprecated
import nest_asyncio
nest_asyncio.apply()  # IMPORTANT


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

        _schema = {_n: (_t, Field(default=None)) for _n, _t in types.items()}
        output_model = create_model(run_func.__name__ + "_Output", **_schema)  # type: ignore

        # TODO async function
        @wraps(run_func)
        async def wrapper(*args, **kwargs) -> Union[BaseModel, Dict, Generator]:
            # TODO 0926
            inputs = input_model(**kwargs)  # type: ignore
            # res = run_func(inputs.model_dump())

            res = run_func(*args, **inputs.__dict__)  # TODO 1114
            # res = run_func(*args, **inputs.model_dump())  # Note: need to take args as it includes `self`
            
            if isinstance(res, Awaitable):
                res = await res

            def _post_run(res: Union[Generator, AsyncGenerator, Dict, Tuple]):
                # TODO 1112
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
            
            return _post_run(res)
        # set up searchable object
        setattr(wrapper, "_input_model", input_model)
        setattr(wrapper, "_output_model", output_model)
        return wrapper

    if run_func:
        # Decorator is called without parens
        return socket_types_decorator(run_func)

    return socket_types_decorator


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
