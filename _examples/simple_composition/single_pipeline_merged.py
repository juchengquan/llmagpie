import asyncio
import time
from llmagpie.base.node import MakeNode, BaseNode
from llmagpie.base.pipeline import BasePipeline 

@MakeNode.from_class(func_name="async_call", outputs=dict(outputs=str))
class EntryNode(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_A")
    
@MakeNode.from_class(func_name="async_call", outputs=dict(outputs=str))
class MiddleNode_B(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_B")
    
@MakeNode.from_class(func_name="async_call", outputs=dict(outputs=str))
class MiddleNode_C1(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_C")

@MakeNode.from_class(func_name="async_call", outputs=dict(outputs=str))
class MiddleNode_C2(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_C")

@MakeNode.from_class(func_name="async_call", outputs=dict(outputs=str))
class MiddleNode_D(BaseNode):
    async def async_call(self, inputs: str, inputs2: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D")


if __name__ == "__main__":
    a = EntryNode(name="AA")
    b1 = MiddleNode_B(name="B1")
    b2 = MiddleNode_B(name="B2")
    c1 = MiddleNode_C1(name="C1")
    c2 = MiddleNode_C2(name="C2")
    d = MiddleNode_D(name="DD")

    pipe = BasePipeline(name="OUTER", nodes=[a, b1, b2, c1, c2, d])

    (a >> "outputs") >> ("inputs" >> b1)
    (b1 >> "outputs") >> ("inputs" >> b2)
    (b2 >> "outputs") >> ("inputs" >> d)

    (a >> "outputs") >> ("inputs" >> c1)
    (c1 >> "outputs") >> ("inputs" >> c2)
    (c2 >> "outputs") >> ("inputs2" >> d)

    pipe.compile()
    print("Finished compiling")

    inputs = {
        "AA.inputs": "Hello"
    }

    async def main():
        response = await pipe.async_invoke(inputs=inputs)
        async for ele in response:
            print(ele)

    asyncio.run(
        main()
    )