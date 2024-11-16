import asyncio
import time
from llmagpie.core.nodes import BaseNode, BaseServiceRetriever
from llmagpie.core.pipeline import MultiHeadPipeline
from llmagpie.core.utilities.wrapper import socket_types


class EntryNode(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name)
    
class MiddleNode_B(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs_1: str, inputs_2: str):
        time.sleep(0.1)
        return dict(outputs=inputs_1 + "@" + self.name + inputs_2)

class MiddleNode_C(BaseNode):
    _max_count_visited = 3

    @socket_types(outputs=str, final_outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        if self.count_visited > self._max_count_visited:
            self.logger.warning("Counter Hits Maximum")
            return dict(final_outputs=inputs + "_FINAL")
        else:
            return dict(outputs=inputs + "@" + self.name + f"_{self.count_visited}")

class MiddleNode_D(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        return dict(outputs=inputs + "@" + self.name + "_D")


if __name__ == "__main__":
    a1 = EntryNode(name="A1")
    a2 = EntryNode(name="A2")
    b = MiddleNode_B(name="B")
    c = MiddleNode_C(name="C")
    d = MiddleNode_D(name="D")
    pipe = MultiHeadPipeline(name="pipeline", nodes=[a2, a1, b, c, d])

    (a1 >> "outputs") >> ("inputs_1" >> b)
    (a2 >> "outputs") >> ("inputs_2" >> b)
    (b >> "outputs") >> ("inputs" >> c)
    (c >> "final_outputs") >> ("inputs" >> d)
    (c >> "outputs") >> ("inputs_2" >> b)

    pipe.compile()
    print("Finished compiling")
    
    inputs = {
        "A2.inputs": "Hello",
        "A1.inputs": "Hello",
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)