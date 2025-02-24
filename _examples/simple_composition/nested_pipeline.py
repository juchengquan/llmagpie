import asyncio
from llmagpie.base.node import MakeNode, BaseNode
from llmagpie.base.pipeline import BasePipeline


@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class EntryNode(BaseNode):
    identifier: str
    
    async def _trigger(self, inputs: str):
        await asyncio.sleep(0.1)
        return dict(outputs=inputs + "@" + self.identifier)
    
@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class EndNode(BaseNode):
    identifier: str
    
    async def _trigger(self, inputs: str):
        self.logger.debug("END!")
        await asyncio.sleep(0.1)
        return dict(outputs=inputs + "@" + self.identifier)


if __name__ == "__main__":
    a = EntryNode(name="A", identifier="A")
    b1 = EntryNode(name="B1", identifier="B1")
    b2 = EntryNode(name="B2", identifier="B2")
    c1 = EntryNode(name="C1", identifier="C1")
    c2 = EntryNode(name="C2", identifier="C2")
    d = EndNode(name="D", identifier="D")

    p_b = BasePipeline(name="BB_PIPELINE", nodes=[b1, b2])
    # (b1 >> "outputs") >> ("inputs" >> b2)
    p_b.add_edge(b1, b2, "outputs", "inputs")
    p_b.compile()
    
    p_c = BasePipeline(name="CC_PIPELINE", nodes=[c1, c2])
    (c1 >> "outputs") >> ("inputs" >> c2)
    p_c.compile()
    
    mid_p = BasePipeline(name="MID", nodes=[p_b, p_c])
    (p_b >> "B2.outputs") >> ("C1.inputs" >> p_c)
    mid_p.compile()

    pipe = BasePipeline(name="NESTED PIPELINE OUTER", nodes=[a, mid_p, d])
    (a >> "outputs") >> ("BB_PIPELINE.B1.inputs" >> mid_p)
    (mid_p >> "CC_PIPELINE.C2.outputs") >> ("inputs" >> d)
    pipe.compile()

    inputs = {
        "A.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)