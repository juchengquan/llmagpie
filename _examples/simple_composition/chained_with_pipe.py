import time
from llmagpie.base.node import MakeNode, BaseNode
from llmagpie.base.pipeline import BasePipeline


@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class EntryNode(BaseNode):
    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_A")

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_B(BaseNode):
    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_B")

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_C(BaseNode):
    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_C")

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_D(BaseNode):
    async def _trigger(self, inputs: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D")


if __name__ == "__main__":
    a = EntryNode(name="AA")
    b1 = MiddleNode_B(name="B1")
    b2 = MiddleNode_B(name="B2")
    c1 = MiddleNode_C(name="C1")
    c2 = MiddleNode_C(name="C2")
    d = MiddleNode_D(name="DD")

    p_b = BasePipeline(name="B_PIPE", nodes=[b1, b2])
    p_c = BasePipeline(name="C_PIPE", nodes=[c1, c2])

    (b1 >> "outputs") >> ("inputs" >> b2)
    p_b.compile()
    (c1 >> "outputs") >> ("inputs" >> c2)
    p_c.compile()

    pipe = BasePipeline(name="OUTER", nodes=[a, p_b, p_c, d])

    (a >> "outputs") >> ("B1.inputs" >> p_b)
    (p_b >> "B2.outputs") >> ("C1.inputs" >> p_c)
    (p_c >> "C2.outputs") >> ("inputs" >> d)

    pipe.compile()
    print("Finished compiling")

    inputs = {
        "AA.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)
