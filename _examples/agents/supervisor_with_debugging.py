"""Supervisor + worker with debug-mode capture and a budget trip.

Demonstrates the four pieces of the observability surface working
together:

1. **RunContext correlation** — log lines and the (would-be) OTel
   spans both carry the same ``run_id``.
2. **OTel GenAI spans** — `agent_span`, `handoff_span`, `chat_span`
   are emitted automatically; this script doesn't set up an exporter,
   but the spans would render in Phoenix / Arize / LangSmith if one
   were registered.
3. **Debug-mode tape capture** — every LLM round-trip is written to
   a per-agent JSONL file under ``./.llmagpie-debug/``.
4. **`format_error`** — when the supervisor's budget trips, the
   exception carries the full :class:`RunContext` and the in-flight
   :class:`DelegationTrace`; ``format_error`` produces a
   human-readable post-mortem.

Run it::

    PYTHONPATH=libs python _examples/agents/supervisor_with_debugging.py

The script intentionally trips ``max_tokens_per_run`` so you can see
both the success-path tape entries AND the post-mortem.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from _mock import MockLLMNode
from llmagpie.experimental.agent import Agent, BudgetExceededError
from llmagpie.experimental.nodes.generators._base import LLMResponse, LLMUsage
from llmagpie.experimental.orchestration import Supervisor
from llmagpie.observability import format_error


def _handoff(worker: str, task: str, call_id: str) -> LLMResponse:
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


async def main() -> None:
    # Send tapes into a tempdir so re-running the example doesn't
    # accumulate files under cwd. Real users would point ``debug_dir``
    # at a project-scoped path (and add it to ``.gitignore``).
    debug_dir = Path(tempfile.mkdtemp(prefix="llmagpie-debug-"))
    print(f"# tapes will land in: {debug_dir}\n")

    researcher = Agent(
        llm=MockLLMNode(name="researcher_llm").configure(
            [
                LLMResponse(
                    content="Sources: arXiv:2312.00752, arXiv:2405.21060.",
                    usage=LLMUsage(
                        prompt_tokens=2000, completion_tokens=2000, total_tokens=4000
                    ),
                ),
            ]
        ),
        model="mock",
        name="researcher",
        debug=True,
        debug_dir=debug_dir,
    )

    writer = Agent(
        llm=MockLLMNode(name="writer_llm").configure(
            [
                LLMResponse(
                    content="Mamba is a selective state-space model …",
                    usage=LLMUsage(
                        prompt_tokens=3000, completion_tokens=3000, total_tokens=6000
                    ),
                ),
            ]
        ),
        model="mock",
        name="writer",
        debug=True,
        debug_dir=debug_dir,
    )

    supervisor = Supervisor(
        llm=MockLLMNode(name="supervisor_llm").configure(
            [
                _handoff("researcher", "Find sources on Mamba SSMs.", "t1"),
                _handoff("writer", "Summarize the findings in two sentences.", "t2"),
                # A third response is scripted but the budget will trip first.
                LLMResponse(
                    content="(unreachable — budget already tripped)",
                    usage=LLMUsage(
                        prompt_tokens=1000, completion_tokens=1000, total_tokens=2000
                    ),
                ),
            ]
        ),
        model="mock",
        system_prompt=(
            "Delegate to `researcher` first, then to `writer`. Each "
            "task description is self-contained — workers don't see "
            "prior conversation."
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
        # 10k token ceiling; researcher (4k) + writer (6k) + supervisor
        # round-trips already nudge past it, so the next round-trip
        # will trip the budget.
        max_tokens_per_run=10_000,
        debug=True,
        debug_dir=debug_dir,
        name="planner",
    )

    try:
        result = await supervisor.run("Write a brief on Mamba SSMs.")
    except BudgetExceededError as exc:
        # The exception now carries a RunContext + DelegationTrace
        # populated by the framework's `attach_context` call. The
        # `format_error` helper renders both so users get a
        # post-mortem without grepping log lines.
        print("# ── post-mortem ──────────────────────────────────────────")
        print(format_error(exc))
        print()

        # The tape files written before the budget tripped are still
        # on disk — point at them so the user can replay the failing
        # exchange against a different model or prompt.
        print("# ── captured tapes ───────────────────────────────────────")
        for tape in sorted(debug_dir.glob("*.jsonl")):
            n = sum(1 for _ in tape.open())
            print(f"  {tape.name:<40} ({n} entries)")
        return

    # Success path — wouldn't fire in this demo, but kept so the
    # script doubles as a template for the non-failing case.
    print(f"final answer: {result.content}")
    print(f"tape file: {result.tape_path}")


if __name__ == "__main__":
    asyncio.run(main())
