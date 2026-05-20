"""Minimal Agent — stateless single-turn responses.

Run with: ``PYTHONPATH=libs python _examples/agents/chatbot.py``

To swap in a real provider:

    from llmagpie.experimental.nodes.generators.anthropic_node import AnthropicChatNode
    llm = AnthropicChatNode(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = "claude-sonnet-4-5"
"""

import asyncio

from _mock import MockLLMNode
from llmagpie.experimental.agent import Agent


async def main() -> None:
    llm = MockLLMNode(name="llm").configure(
        script=[
            "Why hello there! How can I help today?",
            "Sure, the answer is 42.",
        ]
    )

    agent = Agent(
        llm=llm,
        model="mock-model",
        system_prompt="You are concise and friendly.",
    )

    for question in ["hi", "what's the meaning of life?"]:
        result = await agent.run(question)
        print(f"USER: {question}")
        print(f"BOT : {result.content}")
        print(f"      ({result.usage.total_tokens} tokens)\n")


if __name__ == "__main__":
    asyncio.run(main())
