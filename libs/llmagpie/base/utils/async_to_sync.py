from asyncio import AbstractEventLoop
from typing import Any, Generator, AsyncGenerator
from llmagpie.base.utils.thread import AsyncThread


def exec_generator_in_event_loop(async_generator: AsyncGenerator, loop: AbstractEventLoop) -> Generator:
    ait = async_generator.__aiter__()

    async def _get_next() -> tuple[bool, Any]:
        try:
            res = await ait.__anext__()
            done = False
        except StopAsyncIteration:
            res = None
            done = True
        except (BaseException, Exception) as exc:
            res = exc
            done = True
        return done, res
                    
    while True:
        done, result = loop.run_until_complete(_get_next())
        if done:
            if result:
                yield result
            break
        else:
            yield result

def exec_generator_in_separated_thread(async_generator: AsyncGenerator, loop: AbstractEventLoop) -> Generator:
    try:
        thread = AsyncThread(coro=async_generator, loop=loop)
        thread.start()
        thread.join()
        while thread.result:
            if not isinstance(thread.result, (BaseException, Exception)):
                yield thread.result
                thread = AsyncThread(coro=async_generator, loop=loop)
                thread.start()
                thread.join()
            elif isinstance(thread.result, (StopAsyncIteration)):
                break
            else:
                raise thread.result
    except (BaseException, Exception) as exc:
        raise exc

