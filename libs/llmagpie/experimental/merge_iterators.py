import logging
from asyncio import FIRST_COMPLETED, Task, create_task, wait
from collections.abc import AsyncIterable, AsyncIterator, Collection
from typing import TypeVar

_T = TypeVar("_T")
_logger = logging.getLogger(__name__)


async def _await_next(iterator: AsyncIterator[_T]) -> _T:
    return await iterator.__anext__()


def _as_task(iterator: AsyncIterator[_T]) -> Task[_T]:
    return create_task(_await_next(iterator))


async def merge_iterators(iterators: Collection[AsyncIterator[_T]]) -> AsyncIterable[_T]:
    next_tasks = {iterator: _as_task(iterator) for iterator in iterators}
    while next_tasks:
        done, _ = await wait(next_tasks.values(), return_when=FIRST_COMPLETED)
        for task in done:
            iterator = next(it for it, t in next_tasks.items() if t == task)
            try:
                yield task.result()
            except StopAsyncIteration:
                del next_tasks[iterator]
            except Exception as exc:
                # TODO: surface per-iterator errors to the caller instead of swallowing.
                _logger.warning("merge_iterators: iterator raised %r; dropping it.", exc)
                del next_tasks[iterator]
            else:
                next_tasks[iterator] = _as_task(iterator)
