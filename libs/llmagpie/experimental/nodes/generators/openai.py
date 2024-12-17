import base64
import json
import time
from httpx import AsyncClient, Client
from openai import OpenAI

from llmagpie.core.nodes import BaseNode, class_as_node
from llmagpie.core.tools import BaseTool, ToolsNode
# typing
from typing import List, Dict, Any, Optional


@class_as_node(func_name="async_call", outputs=dict(content=str, tool_calls=List[Dict]))
class OpenAIChatCompletionWithToolCall(BaseNode):
    client: OpenAI
    tools_node: Optional[ToolsNode] = None
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        ssl_verify: bool = False,
        timeout: int = 60,
        *args,
        **kwargs,
    ):
        client = OpenAI(
            api_key = api_key,
            base_url = base_url,
            http_client = Client(verify=ssl_verify, timeout=timeout)
        )
        super().__init__(client=client, *args, **kwargs)

    def bind_tools(self, tools: List[BaseTool]):
        self.tools_node = ToolsNode(name=self.name + "_ToolsNode", tools=tools) # TODO
        return self

    async def _single_call(
        self,
        model,
        messages,
        direct_tool_outputs,
    ):
        call_kwargs = dict(
            model=model,
            messages=messages,
            # stream=stream,
        )
        if self.tools_node:
            call_kwargs.update(dict(
                tools=self.tools_node._generate_openai_schema()    
            ))
        response = self.client.chat.completions.create(**call_kwargs)  # type: ignore
        post_response = get_llm_answer(response)

        if self.tools_node:
            post_response["tool_calls"] = (await self.tools_node.async_call_(tool_calls_list=post_response["tool_calls"])).get("tool_calls_list", [])
        elif direct_tool_outputs:
            self.logger.warning("Tools is not binded but `direct_tool_outputs` is set True.... omit")
            direct_tool_outputs = False
        else:
            post_response["tool_calls"] = [] 
        
        return post_response, direct_tool_outputs

    def _add_messages_from_tools(
        self,
        post_response,
        messages,
        direct_tool_outputs
        
    ):
        if direct_tool_outputs:
            return post_response
        
        for ele in post_response["tool_calls"]:
            messages.append({
                "role": post_response["role"],
                "tool_calls": [{
                    "id": ele["id"],
                    "type": ele["type"],
                    "function": ele["function"],
                }]
            })
            messages.append({
                "role": "tool",
                "content": json.dumps(ele["output"]),
                "tool_call_id": ele["id"],
            })
    
    async def async_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        direct_tool_outputs: bool = False,
        # stream: bool = False,
    ):
        post_response, direct_tool_outputs = await self._single_call(model, messages, direct_tool_outputs)
        
        # VLLM: only one tool call for each time
        # compose new call to LLM
        temp_counter = 0
        while (not direct_tool_outputs) and post_response["tool_calls"] and temp_counter < 3:
            print("YIELD: ", post_response)
            yield post_response
            self._add_messages_from_tools(
                post_response, messages, direct_tool_outputs
            )
            post_response, direct_tool_outputs = await self._single_call(model, messages,direct_tool_outputs)
            temp_counter += 1
        yield post_response

@class_as_node(func_name="async_call", outputs=dict(content=str, tool_calls=List[Dict]))
class OpenAIChatCompletionStream(BaseNode):
    client: OpenAI
    tools_node: Optional[ToolsNode] = None

    def __init__(
        self,
        api_key: str,
        base_url: str,
        ssl_verify: bool = False,
        timeout: int = 10,
        *args,
        **kwargs,
    ):
        client = OpenAI(
            api_key = api_key,
            base_url = base_url,
            http_client = Client(verify=ssl_verify, timeout=timeout)
        )

        super().__init__(client=client, *args, **kwargs)

    def bind_tools(self, tools: List[BaseTool]):
        self.tools_node = ToolsNode(tools=tools)
        return self

    async def async_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
    ):
        call_kwargs = dict(
            model=model,
            messages=messages,
            stream=stream,
        )
        if self.tools_node:
            call_kwargs.update(dict(
                tools=self.tools_node._generate_openai_schema()    
            ))
            
        response = self.client.chat.completions.create(**call_kwargs)  # type: ignore
        
        post_response = get_llm_answer_stream(response)
        return post_response
    
def get_llm_answer_stream(response):
    import copy

    _model_name: str = ""
    _id: str = ""
    p_def = {
        "function": {"arguments": "", "name": ""},
        "type": "function",
    }

    func_def = copy.deepcopy(p_def)

    tool_list = []
    res: Dict[str, Any] = {
        "content": ""
    }
    for i, chunk in enumerate(response):
        if not _model_name:
            _model_name = chunk.model
        if not _id:
            _id = chunk.id
        assert _model_name == chunk.model, "Error: model name is not consistent!"
        assert _id == chunk.id, "Error: ID is not consistent!"

        choice = chunk.choices[0]
        role = choice.delta.role
        content = choice.delta.content or ""
        finish_reason = choice.finish_reason

        tool_calls = choice.delta.tool_calls

        res["content"] += content
        # Get role from first chunk
        if role:
            res["role"] = role
            res["id"] = chunk.id
            res["model"] = chunk.model
            continue

        if tool_calls:
            assert len(tool_calls) == 1  # only one object if streaming?
            tool_call = tool_calls[0]

            # By Default omitted as it is always function calls
            if tool_call.id:
                # if func_def already has an id, add the existing tool into list
                if func_def.get("id"):
                    tool_list.append(func_def)
                    func_def = copy.deepcopy(p_def)

                func_def["id"] = tool_call.id
                # func_def["index"] = tool_call.index
                func_def["type"] = tool_call.type # by default is `function`
                func_def["function"]["name"] = tool_call.function.name
            func_def["function"]["arguments"] += tool_call.function.arguments if tool_call.function.arguments else ""

    # check the last `finish_reason`
    if finish_reason not in ["length", "content_filter"]:
        if func_def.get("id"):
            tool_list.append(func_def)

    res["tool_calls"] = tool_list
    res["finish_reson"] = finish_reason

    return res


def get_llm_answer(response):
    res = {}

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    _tool_calls = choice.message.tool_calls if choice.message.tool_calls else []

    tool_list = []
    for i, item in enumerate(_tool_calls):
        _t = item.dict()
        # _t["index"] = i
        tool_list.append(_t)

    return {
        "id": response.id,
        "model": response.model,
        "role": choice.message.role,
        "content": choice.message.content if choice.message.content else "",
        "finish_reason": finish_reason,
        "tool_calls": tool_list
    }