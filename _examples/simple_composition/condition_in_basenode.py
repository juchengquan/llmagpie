import asyncio
import time
from llmagpie.core.nodes import BaseNode, BaseServiceRetriever
from llmagpie.core.pipeline import MultiHeadPipeline
from llmagpie.core.utilities.wrapper import socket_types, conditional
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
        print("self.cond_func at B: ",self.cond_func)
        return dict(outputs=inputs + "@" + self.name + "_B")


class MiddleNode_C(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        print("self.cond_func at C: ",self.cond_func(input_value=inputs))
        return dict(outputs=inputs + "@" + self.name + "_C")

class MiddleNode_E(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_E")

class MiddleNode_D(BaseNode):
    @socket_types(outputs=str)
    async def async_call(self, inputs: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D")


def continue_node(input_value: str):
    """Mock funciton.
    """
    print(f'"Hi" in {input_value}: ', "Hi" in input_value)
    return "Hi" in input_value

cond = continue_node(input_value="Hi")

if __name__ == "__main__":
    a = EntryNode(name="A")
    #b = MiddleNode_B(name="B", cond_func=continue_node,inputs_to_cond={"input_value":"inputs"})
    #b = MiddleNode_B(name="B", cond_func=cond)
    b = MiddleNode_B(name="B")
    c = MiddleNode_C(name="C", cond_func=continue_node, inputs_to_cond={"input_value":"inputs"})
    e1 = MiddleNode_E(name="E1")
    e2 = MiddleNode_E(name="E2")
    d = MiddleNode_D(name="D")
    
    pipe = MultiHeadPipeline(name="OUTER", nodes=[a, b, c, e1, e2, d])

    (a >> "outputs") >> ("inputs" >> b)
    (a >> "outputs") >> ("inputs" >> c)
    (b >> "outputs") >> ("inputs" >> e1)
    (c >> "outputs") >> ("inputs" >> e2)
    (e1 >> "outputs") >> ("inputs" >> d)
    (e2 >> "outputs") >> ("inputs" >> d)

    pipe.compile()
    print("Finished compiling")

    inputs = {
        "A.inputs": "Hello"
    }

    AuxExecutor(pipe, inputs)