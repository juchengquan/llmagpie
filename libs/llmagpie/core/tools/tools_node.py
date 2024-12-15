# from collections import OrderedDict
import json
from uuid import uuid4
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from llmagpie.core.nodes import BaseNode
from pydantic import BaseModel
from llmagpie.core.utilities.wrapper import socket_types
# typing
from ._base import Tool


class ToolsNode(BaseNode):
# class ToolsNode(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed: bool = True
        
    tools: List[Tool]
    tools_with_mapping: Dict[str, Tool]

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

    @socket_types(tool_calls=list)
    async def async_call(self, tool_call_list: List[Dict]):
        with ThreadPoolExecutor(max_workers=4) as executor:
            for _i, ele in enumerate(tool_call_list):
                if ele.get("function", None):
                    function_args = ele["function"]
                    ele["id"] = ele.get("id", uuid4().hex)
                    
                    try:
                        _tool = self.tools_with_mapping[function_args["name"]]
                        args = function_args["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        self.logger.info(f"Running tool: {_tool.name}")
                        future = executor.submit(_tool.run, args)
                        
                    except:
                        future = executor.submit(lambda: Exception("Function argument is wrong"))
                    ele["_f"] = future
             
            _result = [e["_f"].result() if not e["_f"].exception() else e["_f"].exception() for e in tool_call_list]
            for ele, res in zip(tool_call_list, _result):
                ele.update({
                    "output": res if not isinstance(res, Exception) else None,
                    "error": res if isinstance(res, Exception) else None,
                })
                ele.pop("_f")

        return tool_call_list