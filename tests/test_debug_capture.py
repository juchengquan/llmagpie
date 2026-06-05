"""Phase-3 tests: debug-mode runtime capture.

Drives an :class:`Agent` / :class:`Supervisor` with ``debug=True``
and asserts the JSONL tape exists, parses, contains the right
entries, and obeys per-agent isolation (supervisor and worker each
get their own tape file).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from llmagpie.experimental.agent import Agent
from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from llmagpie.experimental.orchestration import Supervisor
from pydantic import Field, PrivateAttr


class MockLLMNode(BaseLLMNode):
    """LLM that replays scripted responses; identical to the one used
    by the OTel test module — duplicated here so the two test files
    can move independently."""

    responses: list[LLMResponse] = Field(default_factory=list)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    async def _complete(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        self._calls.append({"messages": [dict(m) for m in messages]})
        if not self.responses:
            raise RuntimeError("MockLLMNode: ran out of scripted responses")
        return self.responses.pop(0)


def _resp(content: str = "ok", *, tool_calls: list[dict] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        model="gpt-4",
        role="assistant",
        usage=LLMUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


def _read_tape(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Default (debug=False) → no overhead, no tape
# ---------------------------------------------------------------------------


def test_debug_disabled_writes_no_tape(tmp_path: Path):
    llm = MockLLMNode(name="m", responses=[_resp("hi")])
    agent = Agent(llm=llm, model="gpt-4", name="solo", debug_dir=tmp_path)

    result = asyncio.run(agent.run("hello"))

    assert result.tape_path is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Single-agent capture
# ---------------------------------------------------------------------------


def test_debug_enabled_writes_tape_with_request_response(tmp_path: Path):
    llm = MockLLMNode(name="m", responses=[_resp("hi")])
    agent = Agent(
        llm=llm,
        model="gpt-4",
        name="solo",
        debug=True,
        debug_dir=tmp_path,
    )

    result = asyncio.run(agent.run("hello"))

    assert result.tape_path is not None
    assert result.tape_path.exists()
    # Filename contract: <run_id8>__<agent>.jsonl
    assert result.tape_path.name.endswith("__solo.jsonl")
    assert result.run_context is not None
    assert result.tape_path.name.startswith(result.run_context.run_id[:8])

    entries = _read_tape(result.tape_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["agent"] == "solo"
    assert entry["request"]["model"] == "gpt-4"
    assert entry["request"]["messages"] == [{"role": "user", "content": "hello"}]
    assert entry["response"]["content"] == "hi"
    assert entry["response"]["usage"]["total_tokens"] == 8


def test_tape_records_one_entry_per_tool_iteration(tmp_path: Path):
    """An agent that loops once on a tool call should produce two
    tape entries — one per LLM round-trip."""
    from llmagpie.base.node import MakeNode

    @MakeNode.from_function(outputs={"out": str})
    def upper(value: str) -> dict:
        """Uppercase the input."""
        return {"out": value.upper()}

    llm = MockLLMNode(
        name="m",
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "upper", "arguments": '{"value": "abc"}'},
                    }
                ],
                finish_reason="tool_calls",
                model="gpt-4",
                role="assistant",
                usage=LLMUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
            ),
            _resp("done"),
        ],
    )
    agent = Agent(
        llm=llm,
        model="gpt-4",
        tools=[upper],
        name="solo",
        debug=True,
        debug_dir=tmp_path,
    )
    result = asyncio.run(agent.run("go"))

    entries = _read_tape(result.tape_path)  # type: ignore[arg-type]
    assert len(entries) == 2
    # First round has the tool-call response; second has the final text.
    assert entries[0]["response"]["tool_calls"]
    assert entries[1]["response"]["content"] == "done"


# ---------------------------------------------------------------------------
# Supervisor + worker isolation
# ---------------------------------------------------------------------------


def test_supervisor_and_worker_get_separate_tapes_when_both_debug(tmp_path: Path):
    """Worker with its own ``debug=True`` writes to its own tape file;
    supervisor's tape captures only the supervisor's own LLM calls."""
    worker_llm = MockLLMNode(name="wm", responses=[_resp("worker out")])
    worker_agent = Agent(
        llm=worker_llm,
        model="gpt-4",
        name="researcher",
        debug=True,
        debug_dir=tmp_path,
    )

    sup_llm = MockLLMNode(
        name="sm",
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "transfer_to_researcher",
                            "arguments": '{"task": "go"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
                model="gpt-4",
                role="assistant",
                usage=LLMUsage(prompt_tokens=4, completion_tokens=4, total_tokens=8),
            ),
            _resp("brief"),
        ],
    )
    sup = Supervisor(
        llm=sup_llm,
        model="gpt-4",
        workers=[worker_agent.as_worker(name="researcher", description="Find.")],
        name="planner",
        debug=True,
        debug_dir=tmp_path,
    )

    result = asyncio.run(sup.run("hi"))

    # Two tape files: one for the supervisor, one for the worker.
    tapes = sorted(p.name for p in tmp_path.iterdir())
    assert any(n.endswith("__planner.jsonl") for n in tapes), tapes
    assert any(n.endswith("__researcher.jsonl") for n in tapes), tapes

    # Both share the same run_id prefix (workers inherit the supervisor's).
    prefixes = {n.split("__")[0] for n in tapes}
    assert len(prefixes) == 1

    sup_tape = next(p for p in tmp_path.iterdir() if p.name.endswith("__planner.jsonl"))
    worker_tape = next(p for p in tmp_path.iterdir() if p.name.endswith("__researcher.jsonl"))
    sup_entries = _read_tape(sup_tape)
    worker_entries = _read_tape(worker_tape)

    # Supervisor made two LLM round-trips; worker made one.
    assert len(sup_entries) == 2
    assert len(worker_entries) == 1

    # Tape labels match — no cross-contamination.
    assert all(e["agent"] == "planner" for e in sup_entries)
    assert all(e["agent"] == "researcher" for e in worker_entries)

    # SupervisorResult exposes the supervisor's tape_path.
    assert result.tape_path is not None
    assert result.tape_path.name.endswith("__planner.jsonl")


def test_worker_without_debug_writes_to_supervisor_tape(tmp_path: Path):
    """When only the supervisor has ``debug=True``, the worker's
    LLM calls land in the supervisor's tape (the ContextVar capture
    is inherited)."""
    worker_llm = MockLLMNode(name="wm", responses=[_resp("worker out")])
    worker_agent = Agent(llm=worker_llm, model="gpt-4", name="researcher")  # no debug

    sup_llm = MockLLMNode(
        name="sm",
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "transfer_to_researcher",
                            "arguments": '{"task": "go"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
                model="gpt-4",
                role="assistant",
                usage=LLMUsage(prompt_tokens=4, completion_tokens=4, total_tokens=8),
            ),
            _resp("brief"),
        ],
    )
    sup = Supervisor(
        llm=sup_llm,
        model="gpt-4",
        workers=[worker_agent.as_worker(name="researcher", description="Find.")],
        name="planner",
        debug=True,
        debug_dir=tmp_path,
    )

    asyncio.run(sup.run("hi"))

    # Only the supervisor's tape exists.
    tapes = list(tmp_path.iterdir())
    assert len(tapes) == 1
    assert tapes[0].name.endswith("__planner.jsonl")

    entries = _read_tape(tapes[0])
    # 2 supervisor round-trips + 1 worker round-trip = 3 entries.
    assert len(entries) == 3
    # Worker's entry inherited the supervisor's tape's agent_label.
    assert all(e["agent"] == "planner" for e in entries)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_default_debug_dir_is_cwd_relative(tmp_path: Path, monkeypatch):
    """When ``debug_dir`` is unset, the tape lands under
    ``./.llmagpie-debug/`` (cwd-relative). Use chdir to keep the
    test sandboxed."""
    monkeypatch.chdir(tmp_path)
    llm = MockLLMNode(name="m", responses=[_resp("hi")])
    agent = Agent(llm=llm, model="gpt-4", name="solo", debug=True)

    result = asyncio.run(agent.run("hello"))

    assert result.tape_path is not None
    assert (tmp_path / ".llmagpie-debug").exists()
    assert result.tape_path.parent.resolve() == (tmp_path / ".llmagpie-debug").resolve()


def test_resolve_debug_path_sanitizes_agent_label(tmp_path: Path):
    from llmagpie.observability import resolve_debug_path

    out = resolve_debug_path(
        debug_dir=tmp_path,
        run_id="abcdef1234",
        agent_label="my/weird name:agent",
    )
    # Slashes / spaces / colons replaced with underscores.
    assert "/" not in out.name
    assert " " not in out.name
    assert ":" not in out.name
    assert out.name == "abcdef12__my_weird_name_agent.jsonl"
