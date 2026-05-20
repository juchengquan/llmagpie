from asyncio import AbstractEventLoop
from typing import Union, AsyncGenerator, Awaitable
from inspect import isasyncgen, isawaitable
from threading import Thread

class AsyncThread(Thread):
    def __init__(self, coro: Union[Awaitable, AsyncGenerator], loop: AbstractEventLoop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._coro = coro
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
            if isawaitable(self._coro):
                self.result = self._loop.run_until_complete( self._coro )
            elif isasyncgen(self._coro):
                self.result = self._loop.run_until_complete( self._coro.__anext__() )
            else:
                raise TypeError("Input coro type is wrong.")
                
        except StopAsyncIteration:
            self.result = None
        except Exception as exc:
            # Thread.run swallows raises into stderr; the bridge in
            # async_to_sync reads `result` to detect failure, so store the
            # exception there. KeyboardInterrupt / SystemExit propagate.
            self.result = exc
