"""Pretty-print an llmagpie debug-mode tape.

Run::

    python -m llmagpie.observability.tape path/to/run.jsonl
    python -m llmagpie.observability.tape ./.llmagpie-debug/*.jsonl --summary

A tape is the JSONL file written by ``Agent(debug=True)`` /
``Supervisor(debug=True)``. Each line is a JSON object with
``{timestamp, agent, request{model, messages, kwargs}, response{...,
usage}}``. This module renders the file as a chat-style transcript
with token counts so users can inspect a run without writing Python.

The rendering is best-effort: tapes from older format versions, or
tapes that ran through a non-default provider with unusual fields,
still render the parts the CLI recognizes. Use ``--raw`` to fall
back to a pretty-printed JSON dump for anything the prose layout
can't reach.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO


class _Style:
    """ANSI escape codes, gated on whether the output sink is a tty.

    Holding them on one object keeps every render site terse and
    means the colour-on/off toggle flows from a single place.
    """

    __slots__ = ("bold", "cyan", "dim", "green", "magenta", "red", "reset", "yellow")

    def __init__(self, enabled: bool) -> None:
        e = enabled
        self.dim = "\033[2m" if e else ""
        self.bold = "\033[1m" if e else ""
        self.cyan = "\033[36m" if e else ""
        self.yellow = "\033[33m" if e else ""
        self.magenta = "\033[35m" if e else ""
        self.green = "\033[32m" if e else ""
        self.red = "\033[31m" if e else ""
        self.reset = "\033[0m" if e else ""


def _truncate(s: str, max_len: int = 200) -> str:
    """Collapse whitespace and cap length so a noisy assistant turn
    doesn't dominate the transcript."""
    s = " ".join(s.split())
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _load_tape(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _summarize(entries: list[dict[str, Any]]) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` summed across
    every entry. Missing usage fields count as zero."""
    prompt = 0
    completion = 0
    for e in entries:
        usage = (e.get("response") or {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens", 0) or 0)
        completion += int(usage.get("completion_tokens", 0) or 0)
    return prompt, completion


def _stringify_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, default=str)
    except (TypeError, ValueError):
        return repr(args)


def _render_message(msg: dict[str, Any], style: _Style) -> list[str]:
    role = msg.get("role", "?")
    content = msg.get("content")
    tool_calls = msg.get("tool_calls") or []
    tool_call_id = msg.get("tool_call_id")

    role_color = {
        "system": style.dim,
        "user": style.green,
        "assistant": style.cyan,
        "tool": style.yellow,
    }.get(role, "")
    head = f"{role_color}[{role}]{style.reset}"
    if tool_call_id:
        head += f" {style.dim}(call={tool_call_id}){style.reset}"

    out: list[str] = []
    if content:
        out.append(f"  {head} {_truncate(str(content))}")
    elif tool_calls:
        out.append(f"  {head} (tool_calls)")
    else:
        out.append(f"  {head}")

    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "?")
        args = _truncate(_stringify_args(fn.get("arguments")), 80)
        out.append(f"      {style.magenta}→{style.reset} {name}({args})")
    return out


def _render_response(resp: dict[str, Any], style: _Style) -> list[str]:
    out: list[str] = []
    for tc in resp.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name", "?")
        args = _truncate(_stringify_args(fn.get("arguments")), 120)
        out.append(f"  {style.magenta}→ tool_call:{style.reset} {name}({args})")
    content = resp.get("content")
    if content:
        out.append(f"  {style.cyan}[assistant]{style.reset} {_truncate(str(content))}")
    finish = resp.get("finish_reason")
    if finish:
        out.append(f"  {style.dim}finish_reason={finish}{style.reset}")
    return out


def _render_entry(entry: dict[str, Any], n: int, style: _Style) -> str:
    ts = entry.get("timestamp", "?")
    agent = entry.get("agent") or "?"
    req = entry.get("request") or {}
    resp = entry.get("response") or {}
    model = req.get("model", "?")
    messages = req.get("messages") or []
    usage = resp.get("usage") or {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)

    lines = [
        f"{style.bold}═══ entry {n} · {ts} · {agent} ═══{style.reset}",
        f"{style.dim}model:{style.reset} {model}",
        f"{style.dim}messages:{style.reset}",
    ]
    for m in messages:
        lines.extend(_render_message(m, style))
    lines.append(f"{style.dim}response:{style.reset}")
    lines.extend(_render_response(resp, style))
    lines.append(f"{style.dim}usage:{style.reset} {pt} prompt + {ct} completion = {pt + ct} tokens")
    return "\n".join(lines)


def _render_tape(
    path: Path,
    style: _Style,
    *,
    summary_only: bool = False,
    raw: bool = False,
) -> str:
    entries = _load_tape(path)
    prompt, completion = _summarize(entries)
    total = prompt + completion
    n = len(entries)

    header_top = f"{style.bold}# {path}{style.reset}"
    header_stats = (
        f"{style.dim}# {n} entries · {prompt} prompt + {completion} completion "
        f"= {total} tokens{style.reset}"
    )

    if n == 0:
        return f"{header_top}\n{style.dim}# (empty tape){style.reset}"

    if summary_only:
        return f"{header_top}\n{header_stats}"

    if raw:
        body = "\n\n".join(json.dumps(e, indent=2, default=str) for e in entries)
    else:
        body = "\n\n".join(_render_entry(e, i + 1, style) for i, e in enumerate(entries))

    footer = (
        f"{style.dim}# total: {prompt} prompt + {completion} completion "
        f"= {total} tokens · {n} entries{style.reset}"
    )
    return f"{header_top}\n{header_stats}\n\n{body}\n\n{footer}"


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    """Console entry point. Returns the process exit code so callers
    embedding the CLI in a test (or a parent runner) can branch on
    it without forking."""
    parser = argparse.ArgumentParser(
        prog="python -m llmagpie.observability.tape",
        description="Pretty-print one or more llmagpie debug-mode tape files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSONL tape file(s) written by Agent/Supervisor with debug=True.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only print the per-tape header (n entries + token totals).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Print each entry as a pretty-printed JSON object instead of "
            "the chat transcript layout. Mutually exclusive with --summary."
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color escapes (auto-disabled when stdout is not a tty).",
    )
    args = parser.parse_args(argv)

    if args.summary and args.raw:
        parser.error("--summary and --raw are mutually exclusive")

    stream = out if out is not None else sys.stdout
    use_color = not args.no_color and bool(getattr(stream, "isatty", lambda: False)())
    style = _Style(use_color)

    missing = [p for p in args.paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: tape not found: {p}", file=sys.stderr)
        return 2

    for i, path in enumerate(args.paths):
        if i > 0:
            print("", file=stream)
        print(
            _render_tape(path, style, summary_only=args.summary, raw=args.raw),
            file=stream,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
