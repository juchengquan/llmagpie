from asyncio import FIRST_COMPLETED, Task, create_task, wait
from typing import AsyncIterable, AsyncIterator, Collection, TypeVar


_T = TypeVar("_T")


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
            except Exception:
                pass
            else:
                next_tasks[iterator] = _as_task(iterator)