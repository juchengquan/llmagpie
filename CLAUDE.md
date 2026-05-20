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
- `libs/llmagpie/core/opentelemetry/` — optional OTEL decorator. Becomes a
  no-op `EmptyWrapDecorator` if `OTEL_COLLECTOR_ENDPOINT` is unset or
  `opentelemetry` is not installed.
- `libs/llmagpie/experimental/` — not part of the public surface. Anything
  here can change. Includes `_chroma.py`, `sqlite_db/`, `nodes/generators/`.
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
