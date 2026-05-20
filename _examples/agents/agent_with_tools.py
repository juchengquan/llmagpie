"""Agent that calls tools, with a budget ceiling and a semantic stop.

Demonstrates four building blocks composed via Agent:

- Tool dispatch (the ``calculator`` tool below).
- ``max_tokens_per_run`` budget enforcement.
- A ``stop_on_content_match`` condition that ends the loop when the
  LLM emits "FINAL" in its content.
- Token + cost reporting at the end.

Run with: ``PYTHONPATH=libs python _examples/agents/agent_with_tools.py``
"""

import asyncio

from _mock import MockLLMNode
from llmagpie.base.node import MakeNode
from llmagpie.experimental.agent import Agent
from llmagpie.experimental.nodes.generators._base import LLMResponse, LLMUsage
from llmagpie.experimental.nodes.generators.stop import stop_on_content_match


@MakeNode.from_function(name="calculator", outputs={"result": str})
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression (digits + - * /)."""
    # eval-with-bare-builtins is FINE for a demo, never for real input.
    allowed = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed):
        return "ERROR: only basic arithmetic"
    return str(eval(expression))


async def main() -> None:
    # Two scripted LLM turns: the first asks for a tool call; the
    # second responds with the final answer plus the "FINAL" sentinel
    # that triggers our stop_condition.
    llm = MockLLMNode(name="llm").configure(
        script=[
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "21 * 2"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
                usage=LLMUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
            ),
            LLMResponse(
                content="The answer is 42. FINAL",
                usage=LLMUsage(prompt_tokens=40, completion_tokens=10, total_tokens=50),
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        model="mock-model",
        system_prompt="Use the calculator tool. Reply with 'FINAL' once done.",
        tools=[calculator],
        stop_condition=stop_on_content_match(r"\bFINAL\b"),
        # Generous ceiling for the demo; in practice set it tightly.
        max_tokens_per_run=1000,
        cost_per_1k_tokens={"prompt": 0.003, "completion": 0.015},
    )

    result = await agent.run("What is 21 * 2?")

    print("USER: What is 21 * 2?")
    print(f"BOT : {result.content}")
    print(f"  -> tokens used:  {result.usage.total_tokens}")
    print(f"  -> est. cost:    ${agent.cost_of(result.usage):.5f}")
    print(f"  -> tool calls remaining at terminal turn: {len(result.tool_calls)}")


if __name__ == "__main__":
    asyncio.run(main())
