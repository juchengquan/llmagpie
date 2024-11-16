from llmagpie.experimental.nodes.generators.openai import OpenAIChatGenerator


api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRfYXBwbGljYXRpb24iOiJjcWp1X3Rlc3QiLCJzZXJ2ZXJfYXBwbGljYXRpb24iOiJvY2JjLWxsbS1wbGF0Zm9ybSIsImNsaWVudF9lbnYiOiJQUkVQUk9EIiwidG9rZW5fZXhwaXJ5X2RhdGVfdGltZSI6IjIwOTktMTItMzEgMjM6NTk6NTkiLCJ0b2tlbl9jcmVhdGlvbl91dGNfdW5peF90aW1lc3RhbXAiOjE3Mjg2MzQ3MDcsInNhbHRfcmFuZG9tX3N0ciI6IldwT09iIn0.L5XFBa8dlw8urijTfPgEm7z0f55R3uB7bzFanKx0Doo"
url = "https://ocbc-llm-platform.ml-3ab7a488-2a6.apps.apps.prod7.ocbc.com/model/llama-70b/openai/v1"

   
llm = OpenAIChatGenerator(
    name="openai",
    api_key=api_key,
    base_url=url,
)

res = llm.async_invoke(
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