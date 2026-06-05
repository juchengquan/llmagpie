import time

from llmagpie import BaseNode, BasePipeline, MakeNode


@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class EntryNode(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_A")

@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class MiddleNode_B(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_B")

@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class MiddleNode_C(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_C")

@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class MiddleNode_D(BaseNode):
    async def async_call(self, inputs: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D")


if __name__ == "__main__":
    a = EntryNode(name="A")
    b1 = MiddleNode_B(name="B1")
    b2 = MiddleNode_B(name="B2")
    c1 = MiddleNode_C(name="C1")
    c2 = MiddleNode_C(name="C2")
    d = MiddleNode_D(name="D")

    p_b = BasePipeline(name="B_PIPE", nodes=[b1, b2])
    (b1 >> "outputs") >> ("inputs" >> b2)
    p_b.compile()

    p_c = BasePipeline(name="C_PIPE", nodes=[c1, c2])
    (c1 >> "outputs") >> ("inputs" >> c2)
    p_c.compile()

    mid_p = BasePipeline(name="MID", nodes=[p_b, p_c])
    (p_b >> "B2.outputs") >> ("C1.inputs" >> p_c)
    mid_p.compile()

    pipe = BasePipeline(name="NESTED PIPELINE OUTER", nodes=[a, mid_p, d])
    (a >> "outputs") >> ("B_PIPE.B1.inputs" >> mid_p)
    (mid_p >> "C_PIPE.C2.outputs") >> ("inputs" >> d)
    pipe.compile()

    inputs = {
        "A.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)
