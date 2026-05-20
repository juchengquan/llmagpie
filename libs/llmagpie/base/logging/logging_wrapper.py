from functools import wraps
from inspect import getfullargspec, iscoroutinefunction

from .logging import get_or_create_logger

logger = get_or_create_logger()


def _is_method(func) -> bool:
    try:
        return getfullargspec(func)[0][0] == "self"
    except (IndexError, TypeError):
        return False


def log_output(func):
    """
    Decorator that logs the function's output. Preserves the sync/async nature
    of the wrapped callable.
    """
    is_method = _is_method(func)

    if iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            if is_method:
                logger.info(f"{args[0].name} | output: {result}")
            return result

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if is_method:
            logger.info(f"{args[0].name} | output: {result}")
        return result

    return sync_wrapper
