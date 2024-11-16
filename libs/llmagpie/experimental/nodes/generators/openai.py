import base64
import json
import time
from httpx import AsyncClient, Client
from openai import OpenAI

from llmagpie.core.utilities.wrapper import socket_types
from llmagpie.core.nodes import BaseNode

# typing
from typing import List, Dict
from pydantic import Field


class OpenAIChatGenerator(BaseNode):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        ssl_verify: bool = False,
        timeout: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            api_key = api_key,
            base_url = base_url,
            http_client = Client(verify=ssl_verify, timeout=timeout)
        )

    @socket_types(
        message=str
    )
    async def async_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
        
        def _get_stream_response(_response):
            accu_content = ""
            for ele in _response:
                content = ele.choices[0].delta.content
                if ele.choices[0].delta.content:
                    accu_content += content
                    yield dict(message=accu_content)
            
        if stream:
            return _get_stream_response(response)
        return dict(message=response.choices[0].message.content)
