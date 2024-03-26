import asyncio
from llmagpie.core.node import BaseNode
from llmagpie.core.node_disposable import BaseNodeDisposable
from llmagpie.core.pipeline.single_pipeline import SinglePipeline
from llmagpie.core.function import func_input_validator
from typing import Type, Set, Dict, Any, List

class TestNode(BaseNode):
    output_keys_internal: List = ["output"]
    
    async def call(self, input: str):
        print(input)
        res = self.NodeDataOutput(
            output="OK" + self.name
        )
        # print(res)
        return res

a = TestNode(
    name = "a",
)
b = TestNode(
    name = "b",
)
c = TestNode(
    name = "c",
)


(a | "output")  >> (b | "input")
(a | "output")  >> (c | "input")

c._make_fire()
print(c.graph.nodes)

print("DONE")
inputs = {
    "input": "TEST"
}

res = a.fire(inputs)
print("******")
print(asyncio.run(res))