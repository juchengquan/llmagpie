from asyncio import create_task, Task
from typing import (
    AsyncIterator, TypeVar, 
    # AsyncIterable, AsyncIterator, Collection, TypeVar,
    # Sequence, Dict, Union, Optional, List, Callable,
)

_T = TypeVar("_T")

async def _await_next(iterator: AsyncIterator[_T]) -> _T:
    return await iterator.__anext__()

def _as_task(iterator: AsyncIterator[_T]) -> Task[_T]:
    return create_task(_await_next(iterator))

def make_as_task(iterator) -> Task[_T]:
    return _as_task(iterator) if isinstance(iterator, AsyncIterator) else create_task(iterator)

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