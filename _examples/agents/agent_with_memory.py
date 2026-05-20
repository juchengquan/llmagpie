"""Multi-turn agent with persistent conversation memory.

Each ``run()`` call under the same ``thread_id`` sees the prior
exchange in context. Different threads are isolated.

Run with: ``PYTHONPATH=libs python _examples/agents/agent_with_memory.py``
"""

import asyncio

from _mock import MockLLMNode
from llmagpie.experimental.agent import Agent
from llmagpie.experimental.nodes.generators.memory import InMemoryStore


async def main() -> None:
    llm = MockLLMNode(name="llm").configure(
        script=[
            "Hi Alice! Nice to meet you.",
            "Your name is Alice — you told me a moment ago.",
            "Hi Bob! What can I do for you?",
        ]
    )

    agent = Agent(
        llm=llm,
        model="mock-model",
        system_prompt="You are a polite assistant who remembers names.",
        memory_store=InMemoryStore(),
    )

    # Alice's thread.
    r1 = await agent.run("Hi, I'm Alice.", thread_id="alice")
    print("[alice] USER: Hi, I'm Alice.")
    print(f"[alice] BOT : {r1.content}\n")

    r2 = await agent.run("What's my name?", thread_id="alice")
    print("[alice] USER: What's my name?")
    print(f"[alice] BOT : {r2.content}\n")

    # Bob's thread starts fresh — no Alice context.
    r3 = await agent.run("Hi, I'm Bob.", thread_id="bob")
    print("[bob]   USER: Hi, I'm Bob.")
    print(f"[bob]   BOT : {r3.content}\n")

    # Reset Alice's thread.
    await agent.clear_history("alice")
    print("[alice] (history cleared)\n")


if __name__ == "__main__":
    asyncio.run(main())
