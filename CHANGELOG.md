# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Token streaming** — `BaseLLMNode.stream_complete()` async-generator
  hook + `StreamChunk` type for incremental updates +
  `BaseLLMNode.collect_stream()` reducer for assembling chunks back
  into an `LLMResponse`. `OllamaChatNode` and `OpenAIChatNode`
  implement it; `Agent.stream(user_message, ...)` yields chunks from
  a single underlying call and (when memory is attached) persists
  the assembled exchange after the stream completes.
- **`OpenAIChatNode`** (`experimental/nodes/generators/openai_node.py`)
  — the recommended OpenAI integration built on `BaseLLMNode`. Composes
  with `Agent`, `MemoryNode`, `CachedLLMNode`, structured outputs, and
  streaming. The legacy `OpenAIChatCompletionWithToolCall` stays in
  place for backwards compatibility.
- **`RecordReplayLLMNode`** (`experimental/nodes/generators/record_replay.py`)
  — wrap any `BaseLLMNode` to record real provider exchanges to a
  JSON-lines tape, then replay deterministically in CI. Three modes
  (`replay` / `record` / `auto`). Raises `TapeMissError` with an
  actionable request preview when a replay test drifts.
- **Budget enforcement on `Agent`** — `max_tokens_per_run`,
  `max_cost_per_run`, `cost_per_1k_tokens` price table, and
  `BudgetExceededError`. Checked after every provider round-trip so
  tool-call loops can't silently overspend. `Agent.cost_of(usage)`
  is exposed for arbitrary `LLMUsage` aggregations.
- **Semantic stop conditions** — `BaseLLMNode.stop_condition` plumbed
  through the tool-call loop, with factory helpers in
  `experimental/nodes/generators/stop.py`: `stop_on_content_match`,
  `stop_on_tool_name`, `stop_on_finish_reason`, and `any_of`.
- Three runnable example scripts under `_examples/agents/` (chatbot,
  multi-turn memory, tools+budget+stop+cost reporting), wired into
  the test suite via the example-discovery runner.
- `Agent` (`experimental/agent.py`) — high-level wrapper that composes
  `BaseLLMNode` with optional memory, cache, tools, and
  structured-output validation into a single `run(user_message, ...)`
  entry point. Returns an `AgentResult` with content, tool calls,
  cumulative token usage (summed across tool-call rounds), parsed
  schema instance (when `response_schema` is set), and the raw
  `LLMResponse`. Composition order: tools -> memory -> cache -> raw
  provider, so the cache key includes loaded history and is stable
  across processes. Includes `clear_history(thread_id)` for resetting
  per-thread memory. 12 regression tests covering plain runs, memory
  persistence + thread isolation, cache short-circuits, tool-call
  loops, structured-output self-repair, custom params, and
  zero-yield provider misuse.
- `BaseConnectable` exported at the top level
  (`from llmagpie import BaseConnectable`) so users can type-hint
  "any connectable" without reaching into the internal namespace. Added
  to `llmagpie.__all__` and `llmagpie.base.__all__`.
- GitHub issue templates (bug report YAML form, feature request,
  discussions link) and a PR template with the local-check checklist.
- CodeQL security scanning workflow (`security-and-quality` queries,
  weekly schedule + push/PR triggers).
- GitHub Actions CI: lint (ruff), typecheck (mypy), test matrix (Python
  3.12 / 3.13), coverage upload, and wheel/sdist build smoke test.
- Coverage threshold: `[tool.coverage.report] fail_under = 90`. Pytest
  exits non-zero if total branch coverage drops below the threshold.
  `[tool.coverage.run] omit` now excludes `core/opentelemetry/*` —
  the OTEL decorator's real branches only fire with a configured
  collector, which CI doesn't run.
- Subprocess coverage merging in `tests/test_examples.py`: examples
  run under `coverage run --parallel-mode` when pytest-cov is active,
  so the integration runs contribute to the same report. Pushed
  measured coverage from 60% → 77%.
- `.pre-commit-config.yaml` with ruff check + format and the standard
  pre-commit-hooks (trailing whitespace, EOF newline, YAML/TOML lint,
  merge-conflict marker check, 500 KB file-size cap). `pre-commit`
  added to the `dev` dep group.
- `.github/dependabot.yml`: weekly dependency-update PRs for both
  Python deps (uv ecosystem) and GitHub Actions versions. Grouped:
  opentelemetry-* in one PR, types-*/`*-stubs` in another.
- Docstrings for the five most user-facing entry points:
  `BaseConnectable.invoke`, `BaseConnectable.async_invoke`,
  `BaseConnectable.precheck`, `BaseConnectable.clean_states`,
  `BasePipeline.compile`, `BasePipeline.add_edge`.
- `pytest-cov` and `mypy` to the `dev` dependency group; matching
  `[tool.coverage.*]` and `[tool.mypy]` configuration in
  `pyproject.toml`.
- `tests/` directory: unit tests for per-instance state, log_output
  sync/async preservation, exception propagation in the async/sync
  bridge, MakeNode round-trip, precheck contracts, loop cap,
  async_invoke cleanup ordering, ToolsNode, and compile guards.
  Subprocess integration runner over `_examples/simple_composition/*`.
- README "Patterns" table mapping use-case → example file.
- CLAUDE.md orientation document for future agents.
- CONTRIBUTING.md.
- New optional dependency extras: `openai` (was previously imported but
  never declared) and explicit `>=1.0` floor on `chromadb` (validated
  live against chromadb 1.5.9).

### Changed
- **Packaging:** migrated from poetry to `uv` + PEP 621. Build backend
  is now `hatchling`; `uv.lock` is checked in for reproducible
  installs. Old `pyproject_setup_legacy.toml` and `MANIFEST.in`
  removed.
- **Minimum Python: 3.12** (was previously 3.10, then 3.11).
- **Dependency bumps:** networkx ^3.6, pydantic ^2.8 (now uncapped after
  the ConfigDict migration), httpx ^0.28, opentelemetry-* ^1.30,
  sqlalchemy ^2.0.30, wrapt allows 2.x.
- **Pydantic v2 migration:** all `class Config:` blocks replaced with
  `model_config = ConfigDict(...)`. `_SchemaConfig` (plain class) is
  now a `ConfigDict` dict literal; this was the import-time blocker
  for Pydantic >=2.11.
- **Validation errors are now real exceptions, not asserts.** Public
  contracts that survive `python -O`:
  - `RuntimeError` for pipeline state ("not compiled" / "already
    compiled" / "already bound").
  - `ValueError` for malformed user input (schema mismatch, missing
    required keys, DAG with no head/tail nodes).
- `BaseConnectable.async_invoke` no longer cleans state pre-iteration;
  cleanup now runs in the returned generator's `finally`.
- Library catches `except Exception`, not `BaseException` —
  `KeyboardInterrupt` / `SystemExit` propagate through the outer
  `finally` blocks for cleanup but are no longer logged as
  "framework errors".
- `BaseConnectable.__init__` and `BaseConnectDisposable.__init__` no
  longer overwrite a caller-supplied `logger=` kwarg.
- `cond_func(...) == False` rewritten as `not cond_func(...)`.
- `log_output` decorator now preserves the wrapped callable's
  sync/async shape (previously always returned a coroutine).
- `ToolsNode.max_workers` is now a configurable field (default 4).
- Bug-catching exception handlers are now narrow (`KeyError`,
  `ValueError`, `TypeError`, `json.JSONDecodeError`) instead of bare
  `except:`.
- README install and test commands rewritten for `uv sync` / `uv run`,
  with a `pip install -e .` fallback.

### Fixed
- **`async_to_sync.exec_generator_in_event_loop` silently yielded
  iterator exceptions as opaque values** instead of raising them.
  Errors from any node now propagate to the caller as expected
  (regression test in `tests/test_basics.py`).
- **`isinstance(x, Union[A, B])`** (invalid second arg) and a
  `instance(...)` NameError in `core/opentelemetry/_wrapper.py`.
- **`OTEL_ENABLED &= True/False`** rewritten as proper boolean
  assignment.
- **`_input_keys_binded` / `_input_keys_nodes_map`** were class-level
  mutable defaults that pydantic v2 treated as shared class attributes
  — every `BaseConnectable` instance saw the same set/dict. Now
  `PrivateAttr(default_factory=...)`.
- **`OpenAIChatCompletionWithToolCall.num_tool_calls`** is now reset at
  the top of `async_call`. Previously the counter persisted across
  invocations, so the second call short-circuited the tool-call loop.
- **`OpenAIChatCompletionStream` output key typo** `"finish_reson"` →
  `"finish_reason"`.
- **`AppStateBase.value` `unique=True`** dropped — forbade two
  different keys from holding the same value.
- **Mutable default arguments** across the package replaced with
  `Field(default_factory=...)` / `None` + body init.
- **Many naming/spelling**: `is_binded` → `is_bound`,
  `_InterChangableInferface` → `_InterchangeableInterface`,
  "pamatemeters" → "parameters", "Input emprty" → "Input empty",
  `_CRHOMADB_INSTALLED` → `_CHROMADB_INSTALLED`, "binded" → "bound" in
  log messages.
- Pipeline `_validate_root_nodes`: backwards "Only one root is not
  allowed" → "At least one root node is required".
- `MakeNode.from_class` / `from_function` module-level duplicates
  collapsed to delegating thin wrappers.
- **541 ruff lint issues** swept (376 auto-fixed, 32 manual,
  modernization to `list[X]` / `X | None`, `collections.abc` imports,
  unused imports, B026 star-arg-after-kwarg, B904 raise-from, B905
  zip-strict, B028 stacklevel, etc.).
- **Pydantic v1 → v2 method calls:** `item.dict()` → `item.model_dump()`.
- `_error_callback` typed as `NoReturn` so the type checker knows
  control doesn't flow past it.
- Various stale FIXME/TODO/commented-out blocks removed (legacy
  `__main__` smoke test, dead chromadb HttpClient block, `CQJU FIXME
  1009`, etc.).

### Removed
- `pyproject_setup_legacy.toml` (stale setuptools stash with wrong
  license, wrong python version, typos).
- `MANIFEST.in` (setuptools-only; we're on hatchling).
- Dead `MakeNode.base_node` class variable.
- Dead `_max_count_visited` class variable.
- No-op `__init__` overrides on `FunctionSchema` and `BaseNode`.
- Redundant `try/except: raise exc` wrappers.
