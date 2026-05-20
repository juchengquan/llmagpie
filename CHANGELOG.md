# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI: lint (ruff), typecheck (mypy), test matrix (Python
  3.12 / 3.13), coverage upload, and wheel/sdist build smoke test.
- Coverage threshold: `[tool.coverage.report] fail_under = 60`. Pytest
  exits non-zero (and CI fails) if total branch coverage drops below
  60%.
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
