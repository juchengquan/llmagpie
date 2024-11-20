from functools import wraps
from inspect import getfullargspec, isawaitable
from fastapi import HTTPException

from .logging import get_or_create_logger
logger = get_or_create_logger()


def log_output(func):
    """
    Decorator that reports the execution time.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            is_method = getfullargspec(func)[0][0] == 'self'
        except Exception:
            is_method = False

        if is_method:
            _self = args[0]
            result = func(*args, **kwargs)
            if isawaitable(result):
                result = await result
            logger.info(f"{_self.name} | output: {result}")
        else:
            result = func(*args, **kwargs)
            if isawaitable(result):
                result = await result
        return result
    return wrapper


def fastapi_wrapper(func):
    """FastAPI logger wrapper.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        """wrapper
        """
        try:
            result = func(*args, **kwargs)
            if isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=repr(exc))
    return wrapper
