# from collections import OrderedDict
import json
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from pydantic import ConfigDict
from llmagpie.base.node import MakeNode, BaseNode
# typing
from typing import List, Dict


@MakeNode.from_class(func_name="fire", outputs={"tool_calls_list": List[Dict]})
class ToolsNode(BaseNode):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tools: List[BaseNode]
    tools_with_mapping: Dict[str, BaseNode]
    max_workers: int = 4

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

    async def fire(self, tool_calls_list: List[Dict]):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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

                    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        self.logger.warning(f"Tool dispatch failed for {function_args!r}: {exc}")
                        future = executor.submit(lambda exc=exc: Exception(f"Function argument is wrong: {exc}"))
                    ele["_f"] = future
             
            _result = [e["_f"].result() if not e["_f"].exception() else e["_f"].exception() for e in tool_calls_list]
            for ele, res in zip(tool_calls_list, _result):
                ele.update({
                    "output": res if not isinstance(res, Exception) else None,
                    "error": res if isinstance(res, Exception) else None,
                })
                ele.pop("_f")

        return tool_calls_list