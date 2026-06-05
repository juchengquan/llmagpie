# Notes for future agents working on this repo

## Layout

- `libs/llmagpie/` — the package. The wheel/sdist build copies it via
  `[tool.hatch.build.targets.wheel] packages = ["libs/llmagpie"]`. When
  running scripts directly use `PYTHONPATH=libs python ...` or `uv run python`
  inside the synced venv; pytest is already configured via
  `[tool.pytest.ini_options]` `pythonpath`.
- The project uses `uv` + PEP 621 (`[project]` table). Build backend is
  `hatchling`. There's a `uv.lock` checked in for reproducible installs.
- `libs/llmagpie/base/` — public-ish core. Treat as stable.
- `libs/llmagpie/observability/` — public-ish too. Carries the
  `RunContext` spine (a ContextVar-backed correlation object) plus
  `format_error()` / `format_trace()` / `RunContextFilter`, the
  GenAI-semconv span helpers (`agent_span`, `handoff_span`,
  `tool_span`, `chat_span`, `set_llm_attributes`), and the
  debug-mode tape sink (`capture_to`, `current_tape`, `TapeWriter`,
  `resolve_debug_path`). Imported by `base/logging/logging.py` and
  by the experimental `Agent` / `Supervisor` / `WorkerHandle` /
  `BaseLLMNode` / `ToolsNode` entry points. Pure-stdlib + pydantic
  on the core path; `_otel.py` lazy-imports `opentelemetry` and
  gracefully no-ops if the import fails or no tracer provider is
  configured.
- `libs/llmagpie/core/opentelemetry/` — optional OTEL decorator. Becomes a
  no-op `EmptyWrapDecorator` if `OTEL_COLLECTOR_ENDPOINT` is unset or
  `opentelemetry` is not installed. `EmptyWrapDecorator` is the
  identity decorator (returns `func` as-is) so it doesn't pull in
  `wrapt` — that's an `[opentelemetry]`-extra dep, lazy-imported inside
  `WrapDecorator.__call__`.
- `libs/llmagpie/experimental/` — not part of the public surface. Anything
  here can change. Includes `_chroma.py`, `sqlite_db/`, `nodes/generators/`.
  Provider nodes (OpenAI / Anthropic / Ollama) are reference
  implementations gated behind optional extras (`pip install
  llmagpie[openai]` etc.). Production users are expected to bring their
  own client (LangChain, LiteLLM, raw SDK) and wire it as a
  `BaseLLMNode` subclass — that's the supported contract.

## Optional-dep / lazy-import pattern

Anything that imports a third-party SDK at module load time gates the
core install. So everything under `experimental/` follows this pattern:

```python
class FooProviderNode(BaseLLMNode):
    def __init__(self, ...):
        try:
            from foo import FooClient
        except ImportError as e:
            raise ImportError(
                "Could not import `foo`. Install with "
                "`pip install llmagpie[foo]` or `pip install foo`."
            ) from e
        ...
```

Keep the import inside `__init__` (or a small `_build_client` helper),
not at module top. The module must be importable on a minimal install
so things like `from llmagpie.experimental.nodes.generators import ...`
in `__init__.py` don't crash when the user only wanted a different
provider.
- `_examples/simple_composition/` — small runnable demos. They double as
  the integration test corpus via `tests/test_examples.py`.

## The core mental model

```
BaseConnectable           pydantic BaseModel; carries per-session state dicts
├── BaseNode              single unit of work; `async_call_` is the entry
└── BasePipeline          a compiled DAG of connectables
```

- `BaseConnectable` owns three dicts keyed by `session_id`:
  `input_state`, `output_state`, `output_history_state`. Almost every
  bug in this repo's history has been about these dicts (shared across
  sessions, not cleaned up after errors, mutated outside their owner).
- `BasePipeline` wraps a `networkx.DiGraph` (`SingleDAG` in
  `base/pipeline/_dag.py`). Cycle detection is intentionally disabled —
  loop-style pipelines re-enter nodes — see the FIXME there.
- `MakeNode.from_class` / `MakeNode.from_function` synthesize input/output
  Pydantic schemas from a callable's signature. `from_class` rebinds the
  target method onto the class as `async_call_`.
- Connections use the `>>` / `<<` operators. Real usage looks like:
  `(src_node >> "out_key") >> ("in_key" >> dest_node)`. The README's
  quick start has a working minimal example.
- Pipelines must be `.compile()`d before invocation. Compilation freezes
  the input/output schema (`func_schema.external`) and runs DAG
  validation.

## Cross-cutting: `RunContext`

`libs/llmagpie/observability/_context.py` owns a `RunContext` lived in
a `contextvars.ContextVar`. Every public `run()` (`Agent`,
`Supervisor`) and `dispatch()` (`WorkerHandle`) does
`with push(derive(...)): ...`, so:

- Logs carry `run_id` / `agent` / `worker` / `depth` (via
  `RunContextFilter`, attached at the logger level so even pytest's
  `caplog` sees them).
- Framework exceptions (`BudgetExceededError`, etc.) get
  `exc.run_context` populated by `attach_context()` in the entry
  point's `except` block. `format_error(exc)` then renders the
  context + delegation trace.
- Cross-thread propagation: `ToolsNode.fire()` calls
  `contextvars.copy_context()` **per submission** before
  `executor.submit(ctx.run, _tool.run, ...)`. A single context can't
  be entered concurrently, so each parallel tool call needs its own
  copy — caught a regression mid-Phase-1.
- Timezone for log timestamps is `LLMAGPIE_LOG_TZ` (defaults to UTC).
  This replaced the hardcoded `Asia/Singapore` from before.
- GenAI semconv spans: `Agent.run` / `Supervisor.run` open
  `agent_span` (`openinference.span.kind=AGENT`,
  `gen_ai.agent.name=<name>`). `WorkerHandle.dispatch` opens
  `handoff_span` (`openinference.span.kind=CHAIN`,
  `llmagpie.handoff.{source,target,depth,task_preview}`). Each
  provider round-trip goes through `BaseLLMNode._complete_traced`
  which opens a `chat_span` and stamps `gen_ai.usage.input_tokens`
  / `output_tokens` / `gen_ai.request.model` / `gen_ai.system` /
  `gen_ai.response.finish_reasons` from the `LLMResponse`. The
  `system` value is derived by walking the wrapper chain (Memory →
  Cache → … → Provider) to the leaf and reading its
  `provider_name` ClassVar, so the span reports the real provider,
  not "memory". `ToolsNode.fire` opens a `tool_span` per call
  *inside* the worker thread, so the span parents correctly under
  the agent/chat span via the copied OTel context.
- Debug-mode capture: `Agent(debug=True)` /
  `Supervisor(debug=True)` opens a `capture_to(...)` context inside
  `run()` so every LLM round-trip lands in a per-run JSONL tape at
  `<debug_dir>/<run_id8>__<agent_name>.jsonl` (default `debug_dir`
  is `./.llmagpie-debug/`). The sink is a ContextVar — `_complete_traced`
  reads `current_tape()` after each call and appends. Nested
  `capture_to` calls take precedence inside their block (supervisor
  + debug-worker = two tape files; supervisor + plain worker = one
  tape with both agents' calls). Tape path is exposed on
  `AgentResult.tape_path` so callers don't have to compute it.

## Things that have bitten people before

- **Private attrs vs class attrs.** In Pydantic v2, a leading-underscore
  annotation like `_foo: Set = set()` is NOT a model field. It becomes a
  plain class attribute shared by every instance. Use
  `PrivateAttr(default_factory=...)`. `connectable.py` has the canonical
  examples (`_input_keys_binded`, `_input_keys_nodes_map`, `_id`).
- **Mutable default args.** Several pydantic field declarations and plain
  functions had `= []` / `= {}` defaults. Use `Field(default_factory=...)`
  for pydantic, `Optional[...] = None` plus an inside-the-body init for
  plain functions.
- **`isinstance` vs `Union`.** `isinstance(x, Union[A, B])` is not valid;
  use a tuple `isinstance(x, (A, B))`.
- **Pydantic config style.** All models now use
  `model_config = ConfigDict(...)`. Don't go back to the V1 `class Config:`
  block — it triggers `PydanticDeprecatedSince20` warnings and breaks
  with Pydantic >=2.11 when used as `create_model(__config__=SomeClass)`
  (the old `_SchemaConfig` class in `base/node/_schema.py` caused exactly
  this; it's now a `ConfigDict` dict literal).
- **Exceptions in `exec_generator_in_event_loop`.** This sync bridge used
  to catch async-iterator exceptions and yield them as opaque values, so
  callers received `ValueError(...)` instead of having it raised. Don't
  reintroduce that pattern — exceptions must propagate through
  `loop.run_until_complete` and out of the generator. There's a
  regression test in `tests/test_basics.py`.
- **OTEL context detach.** `pipeline/_base.py` has a FIXME about
  `context.detach(token)` raising "ContextVar token was created in a
  different Context" when attach happens inside an async generator. The
  detach is currently commented out; re-enabling requires releasing the
  token in the same context where it was attached (likely via
  `try/finally` around the yields or by switching to span
  context-management).
- **Session isolation.** Per-instance scalar fields like
  `OpenAIChatCompletionWithToolCall.num_tool_calls` leak across
  invocations if not reset. The current fix resets at the top of
  `async_call`; a fuller solution would key the counter by `session_id`.
- **Library catches `except Exception`, not `BaseException`.**
  `KeyboardInterrupt` / `SystemExit` propagate through to the outer
  `finally` blocks in `invoke()` / `async_invoke()` so cleanup still
  runs, but the user can Ctrl+C out without the library
  short-circuiting that as a "node error". The previous code used a
  redundant `(BaseException, Exception)` tuple that was effectively
  `BaseException` and obscured intent.

## Tests

```bash
pytest                    # runs unit + example smoke tests (~20s)
pytest tests/test_basics.py   # just the fast in-process units
```

When fixing a bug worth caring about, add a unit test in
`tests/test_basics.py` first — the example suite catches regressions in
graph behavior but is too coarse for narrow correctness checks.

## Public API surface

- `from llmagpie import BaseNode, BasePipeline, MakeNode, BaseConnectable`
  is the sanctioned import path. `BaseConnectable` is exported so users
  writing functions that operate over "any connectable" can type-hint
  it; direct subclassing is rare but supported.
- `node._id` is an intentional cross-module private attribute. It's
  used as the graph key in `pipeline/_base.py` and `pipeline/_dag.py`.
  Don't rename to `id` (would clash with the Python builtin) and don't
  promote to a public property — users shouldn't need it; if they do,
  they're probably doing something unusual that the operator-overload
  API doesn't cover.

## Don't change in passing

- The `_examples/` leading underscore. It looks unusual for an examples
  directory but the test suite globs it by exact path.
