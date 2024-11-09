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
        return dict(outputs=inputs + "@" + self.name + "_A")
    
class MiddleNode_B(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_B")

class MiddleNode_C(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_C")

class MiddleNode_D(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str, inputs_2: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D" + inputs_2)


if __name__ == "__main__":
    a = EntryNode(name="AA")
    b1 = MiddleNode_B(name="B1")
    b2 = MiddleNode_B(name="B2")
    c1 = MiddleNode_C(name="C1")
    c2 = MiddleNode_C(name="C2")
    d = MiddleNode_D(name="DD")
    
    pipe = MultiHeadPipeline(name="OUTER", nodes=[a, b1, c1, b2, c2, d])

    (a >> "outputs") >> ("inputs" >> b1)
    (b1 >> "outputs") >> ("inputs" >> c1)
    (c1 >> "outputs") >> ("inputs" >> d)
    
    (a >> "outputs") >> ("inputs" >> b2)
    (b2 >> "outputs") >> ("inputs" >> c2)
    (c2 >> "outputs") >> ("inputs_2" >> d)
    pipe.compile()

    inputs = {
        "AA.inputs": "Hello"
    }

    AuxExecutor(pipe, inputs)