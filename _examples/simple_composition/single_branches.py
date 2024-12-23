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
class MiddleNode_C(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        # return dict(outputs=inputs + "@" + self.name + "_C")

        async def func():
            output_val = inputs + "@" + self.name + "_C"
            for i in range(3):
                output_val += str(i)
                yield dict(outputs=output_val)
        return func()

@MakeNode.from_class(func_name="async_call", outputs=dict(outputs=str))
class MiddleNode_D(BaseNode):
    async def async_call(self, inputs: str, inputs_2: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D" + inputs_2)


if __name__ == "__main__":
    a = EntryNode(name="AA")
    b1 = MiddleNode_B(name="B1")
    b2 = MiddleNode_B(name="B2")
    c1 = MiddleNode_C(name="C1")
    c2 = MiddleNode_C(name="C2")
    d = MiddleNode_D(name="DD")

    pipe = BasePipeline(name="OUTER", nodes=[a, b1, c1, b2, c2, d])

    (a >> "outputs") >> ("inputs" >> b1)  # type: ignore
    (b1 >> "outputs") >> ("inputs" >> c1)
    (c1 >> "outputs") >> ("inputs" >> d)

    (a >> "outputs") >> ("inputs" >> b2)
    (b2 >> "outputs") >> ("inputs" >> c2)
    (c2 >> "outputs") >> ("inputs_2" >> d)
    pipe.compile()

    inputs = {
        "AA.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)