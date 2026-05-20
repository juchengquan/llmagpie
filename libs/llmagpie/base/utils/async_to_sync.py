from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator, Generator
from typing import Any

from llmagpie.base.utils.thread import AsyncThread

_DONE = object()  # sentinel for StopAsyncIteration


def exec_generator_in_event_loop(
    async_generator: AsyncGenerator, loop: AbstractEventLoop
) -> Generator:
    ait = async_generator.__aiter__()

    async def _get_next() -> Any:
        try:
            return await ait.__anext__()
        except StopAsyncIteration:
            return _DONE

    while True:
        # Exceptions raised inside the async iterator propagate through
        # `run_until_complete` and out of this generator — they must NOT be
        # converted into yielded values, or callers would receive an
        # exception object as a successful result.
        result = loop.run_until_complete(_get_next())
        if result is _DONE:
            break
        yield result


def exec_generator_in_separated_thread(
    async_generator: AsyncGenerator, loop: AbstractEventLoop
) -> Generator:
    thread = AsyncThread(coro=async_generator, loop=loop)
    thread.start()
    thread.join()
    while thread.result is not None:
        if isinstance(thread.result, BaseException):
            raise thread.result
        yield thread.result
        thread = AsyncThread(coro=async_generator, loop=loop)
        thread.start()
        thread.join()
