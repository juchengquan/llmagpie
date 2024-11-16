import asyncio
import time
from llmagpie.core.nodes import BaseNode, BaseServiceRetriever
from llmagpie.core.pipeline import MultiHeadPipeline
from llmagpie.core.utilities.wrapper import socket_types


class EntryNode(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_A")
    
class MiddleNode_B(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        for i in range(3):
            yield dict(outputs=inputs + "@" + self.name + "_B" + str(i+1)) 
        # return dict(outputs=inputs + "@" + self.name + "_B")

class MiddleNode_C(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_C")


if __name__ == "__main__":
    a = EntryNode(name="A")
    b = MiddleNode_B(name="B")
    c = MiddleNode_C(name="C")
    
    pipe = MultiHeadPipeline(name="OUTER", nodes=[a, b, c])

    (a >> "outputs") >> ("inputs" >> b)
    (b >> "outputs") >> ("inputs" >> c)
    pipe.compile()

    inputs = {
        "A.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)