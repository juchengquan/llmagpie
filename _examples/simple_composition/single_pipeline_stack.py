import time

from llmagpie.base.node import BaseNode, MakeNode
from llmagpie.base.pipeline import BasePipeline


@MakeNode.from_class(func_name="async_call", outputs=dict(entry_outputs=str))
class StartNode(BaseNode):
    async def async_call(self, inputs: str):
        time.sleep(0.1)
        return dict(entry_outputs=inputs + "@" + self.name)

@MakeNode.from_class(func_name="async_call", outputs=dict(middle_output1=str, middle_output2=str, last_outputs=str))
class MiddleNode_B(BaseNode):
    counter: int = 0
    #max_visit_count = 3

    async def async_call(self, initial_inputs: str, other_input1: str="loop data1",other_input2: str="loop data2"):
        time.sleep(0.1)
        self.counter += 1
        if self.counter >= 3:
            print("Counter Hits Maximum")
            return dict(last_outputs=initial_inputs + "_FINAL")
        else:
            print("##Counter: ",self.counter)
            return dict(middle_output1=f'{other_input1}_{self.counter}',middle_output2=f'{other_input2}_{self.counter}')

@MakeNode.from_class(func_name="async_call", outputs=dict(end_outputs=str))
class EndNode(BaseNode):
    async def async_call(self, end_inputs: str):
        time.sleep(0.1)
        return dict(end_outputs=end_inputs + "@" + self.name)


if __name__ == "__main__":
    a = StartNode(name="A")
    b = MiddleNode_B(name="B")
    c = EndNode(name="C")

    pipe = BasePipeline(name="OUTER", nodes=[a, b, c])

    (a >> "entry_outputs") >> ("initial_inputs" >> b)
    (b >> "middle_output1") >> ("other_input1" >> b)
    (b >> "middle_output2") >> ("other_input2" >> b)
    (b >> "last_outputs") >> ("end_inputs" >> c)

    pipe.compile()

    inputs = {
        "A.inputs": "Hello"
    }

    response = pipe.invoke(inputs=inputs)
    for ele in response:
        print(ele)
