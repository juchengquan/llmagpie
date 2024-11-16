import base64
import json
import time
from httpx import AsyncClient, Client
from openai import OpenAI

from llmagpie.core.nodes import BaseServiceRetriever
# from llmagpie.core.sqlite_db.connector import SessionLocal
# from llmagpie.core.sqlite_db.datatype import AppStateBase
from llmagpie.core.utilities.wrapper import socket_types
from llmagpie.core.nodes import BaseNode

# typing
from typing import List, Dict
from pydantic import Field

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRfYXBwbGljYXRpb24iOiJjcWp1X3Rlc3QiLCJzZXJ2ZXJfYXBwbGljYXRpb24iOiJvY2JjLWxsbS1wbGF0Zm9ybSIsImNsaWVudF9lbnYiOiJQUkVQUk9EIiwidG9rZW5fZXhwaXJ5X2RhdGVfdGltZSI6IjIwOTktMTItMzEgMjM6NTk6NTkiLCJ0b2tlbl9jcmVhdGlvbl91dGNfdW5peF90aW1lc3RhbXAiOjE3Mjg2MzQ3MDcsInNhbHRfcmFuZG9tX3N0ciI6IldwT09iIn0.L5XFBa8dlw8urijTfPgEm7z0f55R3uB7bzFanKx0Doo"
url = "https://ocbc-llm-platform.ml-3ab7a488-2a6.apps.apps.prod7.ocbc.com/model/llama-70b/openai/v1"



class OpenAIChatGenerator(BaseNode):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        ssl_verify: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            api_key = api_key,
            base_url = base_url,
            http_client = Client(verify=ssl_verify)
        )

    @socket_types(
        outputs=str
    )
    async def async_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
    ):
        # return dict(outputs="AAA")
        async def func():
            # output_val = inputs + "@" + self.name + "_C"
            for i in range(3):
                # output_val += str(i)
                yield dict(outputs=str(i))
                
        return func()

        # _response = self.client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     logprobs=True,
        #     top_logprobs=1,
        #     stream=True
        # )
        # accu_content = ""
        # for ele in _response:
        #     content = ele.choices[0].delta.content
        #     if ele.choices[0].delta.content:
        #         accu_content += content
        #         yield dict(outputs=accu_content)
        
llm = OpenAIChatGenerator(
    name="openai",
    api_key=api_key,
    base_url=url,
)

res =  llm.async_invoke(
    inputs={
        "model": "Meta-Llama-3.1-70B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are Sexy Samansa, a dirty talker."
            },
            {
                "role": "user",
                "content": "What is the meaning of stonehenge? Give me one-sentence summary."
    
            }
        ]
    }
)

async def main():
    async for ele in res:
        # r = await ele
        print(ele)

import asyncio
asyncio.run(main())