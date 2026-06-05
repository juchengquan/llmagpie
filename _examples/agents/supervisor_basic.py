"""Supervisor / worker — research+writer pipeline.

Three agents:

- ``researcher`` — pretends to find sources for a topic.
- ``writer`` — drafts a summary given the sources.
- ``supervisor`` — coordinates: delegates to researcher first,
  then to writer, then composes the final answer.

All three use the same ``MockLLMNode`` (with scripted responses), so
this script runs without API keys and doubles as an integration test
for the orchestration module.
"""

import asyncio
import json

from _mock import MockLLMNode

from llmagpie.experimental.agent import Agent
from llmagpie.experimental.nodes.generators._base import LLMResponse, LLMUsage
from llmagpie.experimental.orchestration import Supervisor


def _resp_with_tool_call(worker: str, task: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": f"transfer_to_{worker}",
                    "arguments": json.dumps({"task": task}),
                },
            }
        ],
        finish_reason="tool_calls",
        usage=LLMUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
    )


async def main():
    # Build the workers.
    researcher_llm = MockLLMNode(name="researcher_llm").configure(
        [
            LLMResponse(
                content="Sources: arXiv:2312.00752 (Mamba), arXiv:2405.21060 (Mamba-2).",
                usage=LLMUsage(prompt_tokens=15, completion_tokens=20, total_tokens=35),
            ),
        ]
    )
    researcher = Agent(
        llm=researcher_llm, model="mock",
        system_prompt="You find authoritative sources on the topic.",
        name="researcher",
    )

    writer_llm = MockLLMNode(name="writer_llm").configure(
        [
            LLMResponse(
                content=(
                    "Mamba is a selective state-space model that achieves "
                    "Transformer-quality language modeling with linear-time "
                    "inference and constant memory in sequence length."
                ),
                usage=LLMUsage(prompt_tokens=40, completion_tokens=30, total_tokens=70),
            ),
        ]
    )
    writer = Agent(
        llm=writer_llm, model="mock",
        system_prompt="You write technical summaries grounded in the provided sources.",
        name="writer",
    )

    # Build the supervisor.
    supervisor_llm = MockLLMNode(name="supervisor_llm").configure(
        [
            _resp_with_tool_call("researcher", "Find sources on Mamba SSMs.", "t1"),
            _resp_with_tool_call("writer", "Summarize the Mamba findings in 2-3 sentences.", "t2"),
            LLMResponse(
                content=(
                    "Done. Researched and wrote a brief on Mamba SSMs — see worker "
                    "outputs above."
                ),
                usage=LLMUsage(prompt_tokens=50, completion_tokens=15, total_tokens=65),
            ),
        ]
    )

    supervisor = Supervisor(
        llm=supervisor_llm, model="mock",
        system_prompt=(
            "You coordinate a research+writing pipeline. Delegate to `researcher` "
            "first to gather sources, then to `writer` to draft a summary. Each "
            "delegation task description must be self-contained — the workers do "
            "not see prior conversation."
        ),
        workers=[
            researcher.as_worker(
                name="researcher",
                description="Use to gather authoritative sources on a topic.",
            ),
            writer.as_worker(
                name="writer",
                description="Use to draft a summary once research is collected.",
            ),
        ],
        max_delegations=5,
        max_tokens_per_run=10_000,
    )

    result = await supervisor.run("Write a brief on Mamba SSMs.")

    print(f"final answer: {result.content}")
    print(f"total tokens: {result.usage.total_tokens}")
    print(f"workers called: {[wr.worker for wr in result.worker_results]}")
    print("\ndelegation trace:")
    print(result.trace.format())


if __name__ == "__main__":
    asyncio.run(main())
