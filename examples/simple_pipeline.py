import asyncio
from llmagpie.core.node.base_node import BaseNode
from llmagpie.core.pipeline.single_pipeline import SinglePipeline
from typing import Type, Set, Dict, Any, List

class TestNode(BaseNode):
    output_keys_internal: List = ["output"]
    
    async def call(self, input: str):
        res = self.NodeDataOutput(
            output="OK" + self.name
        )
        # print(res)
        return res


if __name__ == "__main__":
    a = TestNode(
        name = "a",
    )
    b = TestNode(
        name = "b",
    )
    c = TestNode(
        name = "c",
    )

    q = SinglePipeline()
    q.add_node(c)  # TODO: single node bugs
    q.add_nodes([a, b])
    print("*"*6)
    print(a.graph)
    print(b.graph)
    print(c.graph)
    print(q.graph)
    print("*"*6)
    
    (a | "output")  >> (b | "input")
    (a | "output")  >> (c | "input")
    
    print(a.graph)
    print(b.graph)
    print(c.graph)
    print(q.graph)

    q.compile()

    print("DONE")
    inputs = {
        "input": "TEST"
    }
    res = q.fire(inputs)
    print("******")
    print(asyncio.run(res))