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
    async def async_call(self, inputs: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D")


if __name__ == "__main__":
    a = EntryNode(name="AA")
    b1 = MiddleNode_B(name="B1")
    b2 = MiddleNode_B(name="B2")
    c1 = MiddleNode_C(name="C1")
    c2 = MiddleNode_C(name="C2")
    d = MiddleNode_D(name="DD")

    p_b = MultiHeadPipeline(name="BB_PIPELINE", nodes=[b1, b2])
    (b1 >> "outputs") >> ("inputs" >> b2)
    p_b.compile()
    
    p_c = MultiHeadPipeline(name="CC_PIPELINE", nodes=[c1, c2])
    (c1 >> "outputs") >> ("inputs" >> c2)
    p_c.compile()
    
    mid_p = MultiHeadPipeline(name="MID", nodes=[p_b, p_c])
    (p_b >> "B2.outputs") >> ("C1.inputs" >> p_c)
    mid_p.compile()

    pipe = MultiHeadPipeline(name="NESTED PIPELINE OUTER", nodes=[a, mid_p, d])
    (a >> "outputs") >> ("BB_PIPELINE.B1.inputs" >> mid_p)
    (mid_p >> "CC_PIPELINE.C2.outputs") >> ("inputs" >> d)
    pipe.compile()

    inputs = {
        "AA.inputs": "Hello"
    }

    AuxExecutor(pipe, inputs)