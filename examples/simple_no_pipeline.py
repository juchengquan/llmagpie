import asyncio
from llmagpie.logging import logger
from llmagpie.core.node.base_node import BaseNode
from typing import Type, Set, Dict, Any, List

class TestNode(BaseNode):
    output_keys_internal: List = ["output"]
    
    async def call(self, input: str):
        res = self.NodeDataOutput(
            output="OK" + self.name
        )
        # print(res)
        return res

class TestNode_2(BaseNode):
    output_keys_internal: List = ["output"]
    
    async def call(self, input: str):
        res = self.NodeDataOutput(
            output="OK" + self.name
        )
        # print(res)
        return res

logger.info("DONE")
    
if __name__ == "__main__":
    a = TestNode(name = "a")
    b = TestNode(
        name = "b",
    )
    c = TestNode_2(
        name = "c",
    )
    print("*"*8)
    print(a.graph)
    print(b.graph)
    print(c.graph)
    print("*"*8)
    (a | ["output"])  >> (b | "input")
    
    (a | "output")  >> (c | "input")

    print("*"*8)
    print(a.graph)
    print(b.graph)
    print(c.graph)
    
    c.compile()
    
    print(a.graph == c.graph)
    
    print(a.graph == b.graph)

    logger.info("DONE")
    inputs = {
        "input": "TEST"
    }

    res = a.fire(inputs)
    print("*"*8)
    print(asyncio.run(res))