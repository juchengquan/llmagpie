# LLMagpie

A lightweight Python framework for composing LLM workflows out of small,
reusable nodes wired into a DAG. Nodes can be plain functions, classes,
or LLM clients; pipelines run them asynchronously, stream results, fan
in/out across branches, loop until a condition, and decompose nested
pipelines as a single node.

> **Framework-first.** The core (`BaseConnectable` / `BaseNode` /
> `BasePipeline`) is the supported public surface and ships with only
> `networkx` + `pydantic` as runtime dependencies. Provider clients,
> vector stores, schedulers, and the OTEL tracer live under
> `experimental/` as reference implementations — opt-in via extras,
> never imported until you touch them. Bring your own (LangChain,
> LiteLLM, the raw SDK) for production.
>
> Status: alpha. The core is stable enough for examples and small
> projects; `experimental/` can change between releases.

## Install

The project uses [`uv`](https://docs.astral.sh/uv/) and PEP 621 metadata.

```bash
uv sync                                  # core only (networkx + pydantic)
uv sync --extra openai                   # OpenAI provider (openai + httpx)
uv sync --extra anthropic                # Anthropic provider
uv sync --extra ollama                   # Ollama provider (just httpx)
uv sync --extra opentelemetry            # OTEL tracing (wrapt + opentelemetry-*)
uv sync --extra exp                      # vector store + sqlite + scheduler
uv sync --group dev                      # dev tooling (pytest, mypy, ruff)
```

Or with pip directly:

```bash
pip install -e .                                  # core
pip install -e ".[openai]"                        # one provider
pip install -e ".[openai,anthropic,ollama]"       # several
pip install -e ".[opentelemetry,exp]"             # everything optional
```

Requires Python 3.12+.

### What's where

| Extra | Pulls in | Use when |
|---|---|---|
| (none) | `networkx`, `pydantic` | You're using the framework with your own LLM client. |
| `openai` | `openai`, `httpx` | `OpenAIChatNode` (recommended) or legacy `OpenAIChatCompletionWithToolCall`. |
| `anthropic` | `anthropic` | `AnthropicChatNode`. |
| `ollama` | `httpx` | `OllamaChatNode` (local models via HTTP, no SDK). |
| `opentelemetry` | `wrapt`, `opentelemetry-*` | You set `OTEL_COLLECTOR_ENDPOINT` and want real spans. |
| `exp` | `chromadb`, `sqlalchemy`, `apscheduler` | Experimental Chroma / sqlite store / scheduler. |

## Quick start

```python
from llmagpie import MakeNode, BaseNode, BasePipeline

@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class Greet(BaseNode):
    async def async_call(self, name: str):
        return {"outputs": f"Hello, {name}!"}

@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class Shout(BaseNode):
    async def async_call(self, outputs: str):
        return {"outputs": outputs.upper()}

greet = Greet(name="greet")
shout = Shout(name="shout")

pipe = BasePipeline(name="hello", nodes=[greet, shout])
# wire output "outputs" of `greet` into input "outputs" of `shout`
(greet >> "outputs") >> ("outputs" >> shout)
pipe.compile()

for state in pipe.invoke(inputs={"greet.name": "world"}):
    print(state.value)
# {'outputs': 'Hello, world!'}
# {'outputs': 'HELLO, WORLD!'}
```

More patterns — branching, looping, nested pipelines, conditional emission,
streaming, multi-input merge — live in [`_examples/simple_composition/`](_examples/simple_composition/).

| Pattern | Example |
|---|---|
| Linear chain, two nodes, single input | `single_pipeline_stack.py` |
| Two upstream nodes merging into one downstream | `single_pipeline_merged.py` |
| Fan-out: one parent into multiple branches that re-join | `single_branches.py`, `chained_branches.py` |
| Multiple parents feeding a single child (multi-input) | `multi_input_with_loop.py` |
| Looping back to an earlier node until a condition holds | `single_input_with_loop.py`, `multi_input_with_loop.py` |
| Conditional emission via `cond_func` on the node | `condition_in_node.py`, `condition_in_basenode.py` |
| Allow a step to skip emission without aborting the pipeline | `single_allow_no_emission.py` |
| Async-iterator streaming output between steps | `single_streaming.py` |
| Composing a pipeline as a node inside an outer pipeline | `nested_pipeline.py`, `chained_with_pipe.py`, `chained_nested_pipe.py` |

## Multi-agent (supervisor / worker)

A `Supervisor` is an `Agent` that delegates to other agents via tool calls:

```python
from llmagpie.experimental.agent import Agent
from llmagpie.experimental.orchestration import Supervisor

researcher = Agent(llm=..., model="...", system_prompt="You find sources.")
writer     = Agent(llm=..., model="...", system_prompt="You draft summaries.")

supervisor = Supervisor(
    llm=..., model="...",
    system_prompt="Delegate to `researcher` then `writer`.",
    workers=[
        researcher.as_worker(name="researcher", description="Find sources on a topic."),
        writer.as_worker(name="writer", description="Draft a summary from sources."),
    ],
    max_delegations=10,
    max_tokens_per_run=50_000,
)

result = await supervisor.run("Write a brief on Mamba SSMs.")
print(result.content)
print(result.usage.total_tokens)          # cumulative across supervisor + workers
print(result.trace.format())              # delegation tree
```

Highlights:

- **Handoff is a tool call.** Workers surface as tools named `transfer_to_<name>`.
  Worker-name and arg validation happen at the dispatch boundary — hallucinated
  names become tool-error messages, not exceptions.
- **Context handoff is configurable per worker.** Default is `task_only` (worker
  sees just the task string + its own system prompt); `task_plus_history` and
  `shared_scratchpad` available via `as_worker(..., context_handoff="...")`.
- **Parallel fan-out.** When the supervisor emits multiple worker tool calls in
  one round, they run concurrently under `asyncio.TaskGroup`, bounded by
  `max_parallel_workers`.
- **Streaming.** `Supervisor.stream(...)` yields `SupervisorChunk` events
  carrying the supervisor's own tokens plus `start`/`end` boundary markers
  around each worker invocation.
- **Budget enforcement** rolls up across nested agents. `BudgetExceededError`
  fires on the next round-trip after the cumulative usage exceeds the cap.

Runnable end-to-end demo with mock LLMs:
[`_examples/agents/supervisor_basic.py`](_examples/agents/supervisor_basic.py).

## Mental model

```
BaseConnectable          (abstract: anything wireable)
├── BaseNode             (a single unit of work)
│   └── @MakeNode.from_class / from_function
└── BasePipeline         (a compiled DAG of connectables)
```

- **Connections** use the `>>` / `<<` operators. `node_a >> "out_key" >> "in_key" << node_b`
  reads "send `out_key` of A into `in_key` of B".
- **State** is per-session, keyed by `session_id`. Each node owns
  `input_state` / `output_state` / `output_history_state` dictionaries
  keyed by session.
- **Execution** is async-first. `invoke()` is a sync wrapper that drives
  the async event loop; `async_invoke()` is the native entry point.
- **Pipelines must be compiled** with `pipe.compile()` before invocation —
  compilation freezes the input/output schema and validates the DAG.

## Tests

```bash
uv sync --group dev
uv run pytest
```

The suite combines unit tests in [`tests/test_basics.py`](tests/test_basics.py)
with example-runner smoke tests in [`tests/test_examples.py`](tests/test_examples.py)
that exercise every file under `_examples/simple_composition/`.

## Debugging & observability

llmagpie carries a small observability surface under
[`libs/llmagpie/observability/`](libs/llmagpie/observability/) that
hooks `Agent` / `Supervisor` / `WorkerHandle` / `BaseLLMNode` /
`ToolsNode` automatically. All four pieces share a single
`RunContext` correlation id so logs, traces, exceptions, and debug
tapes all line up under one `run_id`:

```python
from llmagpie.experimental.agent import Agent, BudgetExceededError
from llmagpie.observability import format_error

agent = Agent(
    llm=my_llm, model="gpt-4o", name="researcher",
    debug=True,                                # write every LLM call to disk
    debug_dir="./.llmagpie-debug",             # default; gitignore it
    max_tokens_per_run=10_000,
)

try:
    result = await agent.run("Find sources on Mamba SSMs.")
except BudgetExceededError as exc:
    print(format_error(exc))                   # rich post-mortem
    raise
```

What you get:

- **Correlation** — every log line carries `run_id=...`, `agent=...`,
  `worker=...`. Configurable timezone via `LLMAGPIE_LOG_TZ` (defaults
  to UTC). JSON-structured logs via `get_or_create_logger(json=True)`
  or `LLMAGPIE_LOG_JSON=1`.
- **OTel GenAI semantic conventions** — `Agent.run` / `Supervisor.run`
  emit `invoke_agent` spans, `WorkerHandle.dispatch` emits `handoff`,
  each provider round-trip emits `chat` with `gen_ai.usage.*` /
  `gen_ai.system` attributes, `ToolsNode.fire` emits `execute_tool`.
  Phoenix / Arize / LangSmith / Langfuse render the tree out of the
  box. No-op when `opentelemetry` isn't installed.
- **Error post-mortems** — `format_error(exc)` renders the
  `RunContext` (run_id, agent, supervisor, worker, depth, thread),
  the in-flight `DelegationTrace`, and well-known extras like the
  budget that was tripped. Works on any exception that bubbled
  through `Agent.run` / `Supervisor.run`.
- **Debug-mode capture** — `debug=True` writes a per-agent JSONL
  tape at `<debug_dir>/<run_id8>__<agent>.jsonl`. Each entry has the
  full request (model, messages, kwargs) and response. Diff-friendly
  for assertions; re-playable into `RecordReplayLLMNode`.
  Supervisor + worker isolation is automatic: when both have
  `debug=True`, each gets its own tape file; when only the
  supervisor does, worker calls land in the supervisor's tape.
  Inspect any tape from the shell:

  ```bash
  python -m llmagpie.observability.tape ./.llmagpie-debug/*.jsonl --summary
  python -m llmagpie.observability.tape ./.llmagpie-debug/<run>__planner.jsonl
  ```

Runnable end-to-end demo:
[`_examples/agents/supervisor_with_debugging.py`](_examples/agents/supervisor_with_debugging.py).

To enable OTLP HTTP export, set `OTEL_COLLECTOR_ENDPOINT` — see
[`libs/llmagpie/core/opentelemetry/_wrapper.py`](libs/llmagpie/core/opentelemetry/_wrapper.py).

## License

GPL-3.0-or-later — see [`LICENSE`](LICENSE).
