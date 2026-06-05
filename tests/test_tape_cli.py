"""Tests for the tape-inspector CLI (``python -m llmagpie.observability.tape``).

Builds tapes by driving a real :class:`Agent` with ``debug=True`` so
the fixtures match what users will actually see on disk.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest
from llmagpie.experimental.agent import Agent
from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage
from llmagpie.observability.tape import _render_tape, _Style, _truncate, main
from pydantic import Field, PrivateAttr


class MockLLMNode(BaseLLMNode):
    responses: list[LLMResponse] = Field(default_factory=list)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    async def _complete(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        self._calls.append({"messages": [dict(m) for m in messages]})
        if not self.responses:
            raise RuntimeError("MockLLMNode: ran out of scripted responses")
        return self.responses.pop(0)


def _make_tape(tmp_path: Path) -> Path:
    """Drive a debug-mode Agent that does one tool-call round + a
    final assistant reply. Returns the tape path."""
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
                usage=LLMUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            ),
            LLMResponse(
                content="All done.",
                finish_reason="stop",
                model="gpt-4",
                role="assistant",
                usage=LLMUsage(prompt_tokens=30, completion_tokens=5, total_tokens=35),
            ),
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
    result = asyncio.run(agent.run("hello"))
    assert result.tape_path is not None
    return result.tape_path


# ---------------------------------------------------------------------------
# Pure-function rendering
# ---------------------------------------------------------------------------


def test_render_tape_includes_header_messages_and_totals(tmp_path: Path):
    path = _make_tape(tmp_path)
    out = _render_tape(path, _Style(enabled=False))

    # Header: filename + total entry count + summed tokens.
    assert str(path) in out
    assert "2 entries" in out
    assert "40 prompt + 7 completion = 47 tokens" in out

    # Per-entry: timestamps + agent label.
    assert "entry 1" in out and "entry 2" in out
    assert "solo" in out

    # Both round-trips rendered: first carries a tool_call, second the assistant text.
    assert "tool_call:" in out
    assert "upper(" in out
    assert "[assistant] All done." in out

    # Footer matches the header.
    assert "total: 40 prompt + 7 completion = 47 tokens · 2 entries" in out


def test_summary_only_skips_per_entry_body(tmp_path: Path):
    path = _make_tape(tmp_path)
    out = _render_tape(path, _Style(enabled=False), summary_only=True)

    assert "2 entries" in out
    # The per-entry header rule should not appear.
    assert "entry 1" not in out
    assert "[assistant]" not in out


def test_raw_mode_dumps_each_entry_as_json(tmp_path: Path):
    path = _make_tape(tmp_path)
    out = _render_tape(path, _Style(enabled=False), raw=True)

    # Two parseable JSON blocks separated by blank lines, in addition to
    # the header/footer prose. Extract every JSON object spanning '{...}'
    # and verify both entries round-trip.
    lines = out.splitlines()
    # Strip prose: '#' header/footer lines aren't part of the JSON.
    json_payload = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#")).strip()
    # The two pretty-printed JSON objects are separated by a blank line.
    blocks = [b for b in json_payload.split("\n\n") if b.strip()]
    assert len(blocks) == 2
    parsed = [json.loads(b) for b in blocks]
    assert all("request" in p and "response" in p for p in parsed)


def test_empty_tape_renders_marker(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    out = _render_tape(empty, _Style(enabled=False))
    assert "(empty tape)" in out


def test_truncate_collapses_whitespace_and_caps_length():
    assert _truncate("a\nb\n\tc") == "a b c"
    assert _truncate("x" * 500, max_len=20).endswith("…")
    assert len(_truncate("x" * 500, max_len=20)) == 20


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_main_writes_transcript_to_passed_stream(tmp_path: Path):
    path = _make_tape(tmp_path)
    buf = io.StringIO()
    rc = main([str(path), "--no-color"], out=buf)
    assert rc == 0
    text = buf.getvalue()
    assert "2 entries" in text
    assert "entry 1" in text and "entry 2" in text


def test_main_summary_flag(tmp_path: Path):
    path = _make_tape(tmp_path)
    buf = io.StringIO()
    rc = main([str(path), "--summary", "--no-color"], out=buf)
    assert rc == 0
    text = buf.getvalue()
    assert "2 entries" in text
    assert "entry 1" not in text


def test_main_returns_nonzero_when_tape_missing(tmp_path: Path, capsys):
    rc = main([str(tmp_path / "nope.jsonl"), "--no-color"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "tape not found" in err


def test_main_rejects_summary_and_raw_together(tmp_path: Path):
    path = _make_tape(tmp_path)
    with pytest.raises(SystemExit):
        main([str(path), "--summary", "--raw"])


def test_main_accepts_multiple_tapes(tmp_path: Path):
    p1 = _make_tape(tmp_path)
    # Build a second tape into a sibling dir so the filenames don't collide.
    p2 = _make_tape(tmp_path / "second")
    buf = io.StringIO()
    rc = main([str(p1), str(p2), "--summary", "--no-color"], out=buf)
    assert rc == 0
    text = buf.getvalue()
    # Both filenames appear in the header section.
    assert str(p1) in text
    assert str(p2) in text


def test_main_disables_color_on_non_tty(tmp_path: Path):
    """When stdout isn't a tty, ANSI codes should be absent even without
    --no-color. ``io.StringIO`` reports ``isatty() == False``."""
    path = _make_tape(tmp_path)
    buf = io.StringIO()
    rc = main([str(path)], out=buf)
    assert rc == 0
    assert "\033[" not in buf.getvalue()
