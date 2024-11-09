import asyncio
import time
from jibberjabber.core.nodes import BaseNode, BaseServiceRetriever
from jibberjabber.core.pipeline import MultiHeadPipeline
from jibberjabber.core.utilities.wrapper import socket_types
# typing
from typing import List
from app_instances._examples.aux_exec import AuxExecutor

class EntryNode(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs)
    
class MiddleNode_B(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs)

class MiddleNode_C(BaseNode):
    _max_count_visited = 3

    @socket_types(outputs=str, final_outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        if self.count_visited > self._max_count_visited:
            print("Counter Hits Maximum")
            return dict(final_outputs=inputs + "_FINAL")
        else:
            return dict(outputs=inputs)
            # return {}

class MiddleNode_D(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        return dict(outputs=inputs)


if __name__ == "__main__":
    a = EntryNode(name="A")
    b = MiddleNode_B(name="B")
    c = MiddleNode_C(name="C")
    d = MiddleNode_D(name="D")
    pipe = MultiHeadPipeline(name="pipeline", nodes=[a,b,c,d])

    (a >> "outputs") >> ("inputs" >> b)
    (b >> "outputs") >> ("inputs" >> c)
    (c >> "outputs") >> ("inputs" >> b)
    (c >> "final_outputs") >> ("inputs" >> d)
    pipe.compile()
    print("Finished compiling")

    inputs = {
        "AA.inputs": "Hello"
    }

    AuxExecutor(pipe, inputs)