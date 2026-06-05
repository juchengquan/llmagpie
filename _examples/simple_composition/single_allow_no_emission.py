import asyncio
import time

from llmagpie import BaseNode, BasePipeline, MakeNode


@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class EntryNode(BaseNode):
    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs)

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_B(BaseNode):
    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs)

@MakeNode.from_class(func_name="_trigger", outputs={"loop_outputs": str, "final_outputs": str})
class MiddleNode_C(BaseNode):
    _max_count_visited = 3

    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        if self.count_visited > self._max_count_visited:
            print("Counter Hits Maximum")
            return dict(final_outputs=inputs + "_FINAL")
        else:
            return dict(loop_outputs=inputs, final_outputs="X")

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_D(BaseNode):
    async def _trigger(self, inputs: str):
        if inputs != "X":
            return dict(outputs=inputs)


if __name__ == "__main__":
    a = EntryNode(name="A")
    b = MiddleNode_B(name="B")
    c = MiddleNode_C(name="C")
    d = MiddleNode_D(name="D")
    pipe = BasePipeline(name="pipeline", nodes=[a,b,c,d])

    (a >> "outputs") >> ("inputs" >> b)
    (b >> "outputs") >> ("inputs" >> c)
    (c >> "loop_outputs") >> ("inputs" >> b)
    (c >> "final_outputs") >> ("inputs" >> d)
    pipe.compile()
    print("Finished compiling")

    inputs = {
        "A.inputs": "Hello"
    }

    async def shell():
        response = pipe.invoke(inputs=inputs)
        for ele in response:
            print(">>>", ele)

    asyncio.run(shell())

    # response = pipe.invoke(inputs=inputs)
    # for ele in response:
    #     print(ele)
