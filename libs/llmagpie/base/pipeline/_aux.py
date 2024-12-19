from asyncio import Task
from typing import (
    AsyncIterator, TypeVar, Coroutine, Union
)
from asyncio import AbstractEventLoop

_T = TypeVar("_T")

async def _await_next(iterator: AsyncIterator[_T]) -> _T:
    return await iterator.__anext__()

def _as_task(iterator: AsyncIterator[_T], loop: AbstractEventLoop) -> Task[_T]:
    return loop.create_task(_await_next(iterator))

def make_as_task(iterator: Union[AsyncIterator, Coroutine], loop: AbstractEventLoop) -> Task:
    return _as_task(iterator, loop) if isinstance(iterator, AsyncIterator) else loop.create_task(iterator)

def decompose_pipeline(dt: dict) -> dict:
    res = {}
    for k, v in dt.items():
        if v is not None:
            if v[-1]["_type"] == "Pipeline":
                res.update({
                    f"{k}.{kk}": vv for kk, vv in decompose_pipeline(v[-1]["value"]).items()
                })
            else:
                res.update({
                    f"{k}.{kk}": vv for kk, vv in v[-1]["value"].items() 
                })
    return res