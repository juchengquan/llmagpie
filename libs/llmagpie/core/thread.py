from asyncio import AbstractEventLoop
from typing import Generator, Iterator, AsyncGenerator
from threading import Thread


class AsyncGenerationThread(Thread):
    def __init__(self, async_generator: AsyncGenerator, loop: AbstractEventLoop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._async_generator = async_generator
        self._loop = loop
        
        self._result = None

    @property
    def result(self):
        return self._result
    
    @result.setter
    def result(self, v):
        self._result = v
    
    def run(self):
        # call the threaded function
        try:
            self.result = self._loop.run_until_complete( self._async_generator.__anext__() )
        except StopAsyncIteration as exc:
            self.result = None
        except (Exception, BaseException) as exc:
            self.result = exc
            raise exc