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
        print("self.cond_func at B: ",self.cond_func)
        return dict(outputs=inputs + "@" + self.name + "_B")

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_C(BaseNode):
    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        print("self.cond_func at C: ",self.cond_func(input_value=inputs))
        return dict(outputs=inputs + "@" + self.name + "_C")

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_E(BaseNode):
    async def _trigger(self, inputs: str):
        time.sleep(0.1)
        return dict(outputs=inputs + "@" + self.name + "_E")

@MakeNode.from_class(func_name="_trigger", outputs={"outputs": str})
class MiddleNode_D(BaseNode):
    async def _trigger(self, inputs: str):
        self.logger.debug("END!")
        return dict(outputs=inputs + "@" + self.name + "_D")


def continue_node(input_value: str):
    """Mock funciton.
    """
    print(f'"Hi" in {input_value}: ', "Hi" in input_value)
    return "Hi" in input_value

cond = continue_node(input_value="Hi")

if __name__ == "__main__":
    a = EntryNode(name="A")
    #b = MiddleNode_B(name="B", cond_func=continue_node,inputs_to_cond={"input_value":"inputs"})
    #b = MiddleNode_B(name="B", cond_func=cond)
    b = MiddleNode_B(name="B")
    c = MiddleNode_C(name="C", cond_func=continue_node, inputs_to_cond={"input_value":"inputs"})
    e1 = MiddleNode_E(name="E1")
    e2 = MiddleNode_E(name="E2")
    d = MiddleNode_D(name="D")
    
    pipe = BasePipeline(name="OUTER", nodes=[a, b, c, e1, e2, d])

    (a >> "outputs") >> ("inputs" >> b)
    (a >> "outputs") >> ("inputs" >> c)
    (b >> "outputs") >> ("inputs" >> e1)
    (c >> "outputs") >> ("inputs" >> e2)
    (e1 >> "outputs") >> ("inputs" >> d)
    (e2 >> "outputs") >> ("inputs" >> d)

    pipe.compile()
    print("Finished compiling")

    inputs = {
        "A.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)