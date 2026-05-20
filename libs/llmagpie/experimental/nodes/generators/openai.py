import json

# typing
from typing import Any

from httpx import Client
from llmagpie.base.node import BaseNode, MakeNode
from llmagpie.base.tools import ToolsNode
from openai import OpenAI


@MakeNode.from_class(func_name="async_call", outputs=dict(content=str, tool_calls=list[dict]))
class OpenAIChatCompletionWithToolCall(BaseNode):
    client: OpenAI
    tools_node: ToolsNode | None = None

    num_tool_calls: int = 0
    max_num_tool_calls: int = 3

    def __init__(
        self,
        api_key: str,
        base_url: str,
        ssl_verify: bool = False,
        timeout: int = 60,
        *args,
        **kwargs,
    ):
        """
        Initialization function.

        Args:
            api_key (str): The API key to be used for the OpenAI conpatible API.
            base_url (str): The base URL for the OpenAI conpatible API.
            ssl_verify (bool, optional): Whether to verify SSL certificates. Defaults to False.
            timeout (int, optional): The timeout for API requests. Defaults to 60.
        """
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=Client(verify=ssl_verify, timeout=timeout),
        )
        kwargs["client"] = client
        super().__init__(*args, **kwargs)

    def bind_tools(self, tools: list[BaseNode]):
        self.tools_node = ToolsNode(name=self.name + "_ToolsNode", tools=tools)
        return self

    async def _single_call(
        self,
        model,
        messages,
    ):
        """
        Args:
            model (str): The name of the OpenAI conpatible model to use.
            messages (List[Dict]): A list of message dictionaries, where each dictionary has at least the keys "role"
                (e.g., "user" or "assistant") and "content" (a string).

        Returns:
            Dict: A dictionary containing the OpenAI response, which includes the "content" of the response
                and a "tool_calls" list if any tools were invoked.

        Raises:
            Exception: If the number of tool calls exceeds the maximum allowed (self.max_num_tool_calls).
        """
        call_kwargs = dict(
            model=model,
            messages=messages,
            # stream=stream,
        )
        if self.tools_node:
            call_kwargs.update(dict(tools=self.tools_node._generate_openai_schema()))
        response = self.client.chat.completions.create(**call_kwargs)
        post_response = _get_llm_answer(response)

        if self.tools_node:
            post_response["tool_calls"] = (
                await self.tools_node.async_call_(tool_calls_list=post_response["tool_calls"])
            ).get("tool_calls_list", [])
        else:
            post_response["tool_calls"] = []

        return post_response

    def _add_messages_from_tools(
        self,
        post_response,
        messages,
    ):
        """
        Adds tool call messages to the conversation history.

        Args:
            post_response (Dict): The response from the LLM containing tool calls.
            messages (List[Dict]): The current conversation history to which tool call messages will be added.

        This method appends two types of messages to the conversation history:
        1. A message representing the tool call request from the LLM.
        2. A message representing the tool's response to the call.
        """
        messages.append(
            {
                "role": post_response["role"],
                "tool_calls": [
                    {
                        "id": ele["id"],
                        "type": ele["type"],
                        "function": ele["function"],
                    }
                    for ele in post_response["tool_calls"]
                ],
            }
        )
        for ele in post_response["tool_calls"]:
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(ele["output"]),
                    "tool_call_id": ele["id"],
                }
            )

        # only for llama <- parallel tool calling
        # for ele in post_response["tool_calls"]:
        #     messages.append({
        #         "role": post_response["role"],
        #         "tool_calls": [{
        #             "id": ele["id"],
        #             "type": ele["type"],
        #             "function": ele["function"],
        #         }]
        #     })
        #     messages.append({
        #         "role": "tool",
        #         "content": json.dumps(ele["output"]),
        #         "tool_call_id": ele["id"],
        #     })

    async def async_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        direct_tool_outputs: bool = False,
        # stream: bool = False,
    ):
        """
        Asynchronously calls the OpenAI chat completion API with optional tool calls.

        Args:
            model (str): The model to use for the chat completion.
            messages (List[Dict[str, Any]]): A list of message dictionaries containing the conversation history.
            direct_tool_outputs (bool, optional): If True, yields the tool outputs directly without further LLM calls.
                                                Defaults to False.

        Yields:
            Dict: A dictionary containing the response content and tool calls. The dictionary has the following keys:
                - content (str): The generated text content.
                - tool_calls (List[Dict]): A list of tool calls, each containing the tool's ID, type, and function details.
        """
        if direct_tool_outputs and not self.tools_node:
            self.logger.warning(
                "Tools are not bound but `direct_tool_outputs` is set True.... omit"
            )
            direct_tool_outputs = False

        # Reset per-invocation counter so the limit applies within a single
        # `async_call`, not cumulatively across invocations of the same node.
        self.num_tool_calls = 0

        post_response = await self._single_call(model, messages)
        if direct_tool_outputs:
            yield post_response
        else:
            # VLLM: only one tool call for each time
            # compose new call to LLM
            while self.num_tool_calls < self.max_num_tool_calls and post_response["tool_calls"]:
                yield post_response
                self._add_messages_from_tools(post_response, messages)
                post_response = await self._single_call(model, messages)
                self.num_tool_calls += 1
            yield post_response


@MakeNode.from_class(func_name="async_call", outputs=dict(content=str, tool_calls=list[dict]))
class OpenAIChatCompletionStream(BaseNode):
    client: OpenAI
    tools_node: ToolsNode | None = None

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
            api_key=api_key,
            base_url=base_url,
            http_client=Client(verify=ssl_verify, timeout=timeout),
        )

        kwargs["client"] = client
        super().__init__(*args, **kwargs)

    def bind_tools(self, tools: list[BaseNode]):
        self.tools_node = ToolsNode(tools=tools)
        return self

    async def async_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
    ):
        call_kwargs = dict(
            model=model,
            messages=messages,
            stream=stream,
        )
        if self.tools_node:
            call_kwargs.update(dict(tools=self.tools_node._generate_openai_schema()))

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
    res: dict[str, Any] = {"content": ""}
    for chunk in response:
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
                func_def["type"] = tool_call.type  # by default is `function`
                func_def["function"]["name"] = tool_call.function.name
            func_def["function"]["arguments"] += (
                tool_call.function.arguments if tool_call.function.arguments else ""
            )

    # check the last `finish_reason`
    if finish_reason not in ["length", "content_filter"]:
        if func_def.get("id"):
            tool_list.append(func_def)

    res["tool_calls"] = tool_list
    res["finish_reason"] = finish_reason

    return res


def _get_llm_answer(response):
    choice = response.choices[0]
    finish_reason = choice.finish_reason
    _tool_calls = choice.message.tool_calls if choice.message.tool_calls else []

    tool_list = []
    for item in _tool_calls:
        _t = item.model_dump()
        # _t["index"] = i
        tool_list.append(_t)

    return {
        "id": response.id,
        "model": response.model,
        "role": choice.message.role,
        "content": choice.message.content if choice.message.content else "",
        "finish_reason": finish_reason,
        "tool_calls": tool_list,
    }
