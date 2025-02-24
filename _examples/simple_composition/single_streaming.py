import asyncio
import time
from llmagpie.base.node import BaseNode, MakeNode
from llmagpie.base.pipeline import BasePipeline


@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class EntryNode(BaseNode):
    identifier: str
    
    async def _trigger(self, inputs: str):
        await asyncio.sleep(0.1)
        return dict(outputs=inputs + "@" + self.identifier)
    
@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class SyncStreamingNode(BaseNode):
    identifier: str
    
    def _trigger(self, inputs: str):
        for i in range(3):
            time.sleep(0.1)
            yield dict(outputs=inputs + "@" + self.identifier + str(i+1)) 


if __name__ == "__main__":
    a = EntryNode(name="A", identifier="A")
    b = SyncStreamingNode(name="B", identifier="B")
    c = EntryNode(name="C", identifier="C")
    
    pipe = BasePipeline(name="OUTER", nodes=[a, b, c])

    (a >> "outputs") >> ("inputs" >> b)
    (b >> "outputs") >> ("inputs" >> c)
    pipe.compile()

    inputs = {
        "A.inputs": "Hello"
    }

    async def main():
        response = pipe.invoke(inputs=inputs)
        for ele in response:
            print(ele)
            
    asyncio.run(
        main()
    )
    
    # response = pipe.invoke(inputs=inputs)
    # for ele in response:
    #     print(ele)