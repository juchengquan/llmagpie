# from collections import OrderedDict
import json
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from llmagpie.core.nodes import BaseNode, class_as_node
# typing
from typing import List, Dict


class BaseTool(BaseNode):
    class Config:
        extra: str = "forbid"
        arbitrary_types_allowed: bool = True

    # name: str
    # """The unique name of the tool that clearly communicates its purpose."""
    # description: str = Field(default="")
    # """Used to tell the model how/when/why to use the tool."""
    # async_call_: Callable
    # """The function that will be executed when the tool is called."""
    # input_model_schema: Type[BaseModel]
    # """The schema for the arguments that the tool accepts."""
    # output_model_schema: Type[BaseModel]
    # async_function:  Optional[Callable] = None
    # """The async function that will be executed when the tool is called."""
    # function_type: Literal["sync", "async"]
    
    # def _generate_description_openai(self):
    #     tool_schema = {
    #         "type": "function",
    #         "function": {
    #             "name": self.name,
    #             "description": self.description,
    #             "parameters": self.input_model_schema.schema()
    #         }
    #     }
    #     return tool_schema


@class_as_node(func_name="bolt", outputs={"tool_calls_list": List[Dict]})
class ToolsNode(BaseNode):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed: bool = True
        
    tools: List[BaseNode]
    tools_with_mapping: Dict[str, BaseNode]

    def __init__(self, *args, **kwargs):
        tools_with_mapping = {
            ele.name: ele for ele in kwargs.get("tools", [])
        }
        
        super().__init__(
            tools_with_mapping=tools_with_mapping,
            *args,
            **kwargs
        )

    def _generate_openai_schema(self):
        return [ele._generate_description_openai() for ele in self.tools]

    def __repr__(self):
        return f"{list(self.tools_with_mapping.keys())}"

    def __str__(self):
        return self.__repr__()

    async def bolt(self, tool_calls_list: List[Dict]):
        with ThreadPoolExecutor(max_workers=4) as executor:
            for _i, ele in enumerate(tool_calls_list):
                if ele.get("function", None):
                    function_args = ele["function"]
                    ele["id"] = ele.get("id", uuid4().hex)
                    
                    try:
                        _tool = self.tools_with_mapping[function_args["name"]]
                        args = function_args["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        
                        self.logger.info(f"Running tool: {_tool.name}")
                        future = executor.submit(_tool.run, **args)
                        
                    except:
                        future = executor.submit(lambda: Exception("Function argument is wrong"))
                    ele["_f"] = future
             
            _result = [e["_f"].result() if not e["_f"].exception() else e["_f"].exception() for e in tool_calls_list]
            for ele, res in zip(tool_calls_list, _result):
                ele.update({
                    "output": res if not isinstance(res, Exception) else None,
                    "error": res if isinstance(res, Exception) else None,
                })
                ele.pop("_f")

        return tool_calls_list