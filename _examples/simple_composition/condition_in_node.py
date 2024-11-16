import asyncio
import time
from llmagpie.core.nodes import BaseNode, BaseServiceRetriever
from llmagpie.core.pipeline import MultiHeadPipeline
from llmagpie.core.utilities.wrapper import socket_types, conditional
# typing
from typing import List

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
        # print("self.cond_func at C: ",self.cond_func(input_value=inputs))
        # return dict(outputs=inputs + "@" + self.name + "_C")
        try:
            if self.cond_func(input_value=inputs)==False:
                print("self.cond_func: ",self.cond_func(input_value=inputs))
                return dict(outputs=inputs + "@" + self.name + "_C")
            else: 
                self.logger.error("Condition not met.")
                return None
        except Exception as err:
            raise err
            #return None

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

def continue_node_default(inputs: str):
    """Mock funciton.
    """
    print(f'"Hi" in {inputs}: ', "Hi" in inputs)
    return "Hi" in inputs


if __name__ == "__main__":
    a = EntryNode(name="A")
    b = MiddleNode_B(name="B")
    # inplicit mapping for conditional functions
    c = MiddleNode_C(name="C", cond_func=continue_node_default)
    # explicit mapping for conditional functions
    # c = MiddleNode_C(name="C", cond_func=continue_node, inputs_to_cond={"input_value":"inputs"})
    e = MiddleNode_E(name="E")
    d = MiddleNode_D(name="D")
    
    pipe = MultiHeadPipeline(name="OUTER", nodes=[a, b, c, e, d])

    (a >> "outputs") >> ("inputs" >> b)
    (a >> "outputs") >> ("inputs" >> c)
    (b >> "outputs") >> ("inputs" >> e)
    (c >> "outputs") >> ("inputs" >> e)
    (e >> "outputs") >> ("inputs" >> d)

    pipe.compile()
    print("Finished compiling")

    inputs = {
        "A.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)