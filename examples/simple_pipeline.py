import asyncio
from llmagpie.core.node import BaseNode
from llmagpie.core.node_disposable import BaseNodeDisposable
from llmagpie.core.pipeline.single_pipeline import SinglePipeline
from llmagpie.core.function import func_input_validator
from typing import Type, Set, Dict, Any, List

class TestNode(BaseNode):
    # input_keys_internal: List[str] = ["input"]
    output_keys_internal: List = ["output"]
    
    # NodeDataInput: Dict = {
    #     input: (str, None)
    # }
    # NodeDataInput: Any = None
    
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

q = SinglePipeline()
q.add_node(c)
q.add_nodes([a, b])

(a | "output")  >> (b | "input")
(a | "output")  >> (c | "input")

q.compile()

print("DONE")
inputs = {
    "input": "TEST"
}
res = q.fire(inputs)
print("******")
print(asyncio.run( res))