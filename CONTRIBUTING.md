# Contributing

## Dev setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev --extra opentelemetry --extra openai
```

This installs the runtime deps, all optional extras you'll likely
touch, and the dev tooling (`pytest`, `pytest-cov`, `ruff`, `mypy`).

The repo uses a `libs/`-prefixed layout. The wheel build (`uv build`)
strips the prefix automatically; for ad-hoc scripts use `uv run` or set
`PYTHONPATH=libs`.

## The local checks CI runs

```bash
uv run ruff check libs/ tests/        # lint
uv run ruff format --check libs/ tests/   # format drift
uv run mypy libs/llmagpie             # type check
uv run pytest                          # 25 tests, ~16s
uv run pytest --cov                    # with coverage
```

`ruff format` (without `--check`) auto-formats. Run it before
committing so CI doesn't bounce on a whitespace nit.

## Tests

- `tests/test_basics.py` — fast in-process unit tests. Add narrow
  regression tests here when you fix a bug.
- `tests/test_examples.py` — parametrised subprocess runner over
  `_examples/simple_composition/*.py`. Acts as integration coverage
  for the pipeline machinery.

Don't rename the `_examples/` directory; the test suite globs it by
exact path.

## Code style

- We follow whatever `ruff format` produces. Line length is 100.
- Type hints are required on public functions (mypy is checked in CI).
- Mutable defaults must use `Field(default_factory=...)` for pydantic
  fields and `Optional[...] = None` + body init for plain functions.
- Don't catch `BaseException`; use `except Exception`. The
  `KeyboardInterrupt` / `SystemExit` propagation policy is documented
  in CLAUDE.md.
- Don't use `assert` for public-API validation — it gets stripped
  under `python -O`. Raise `RuntimeError` (pipeline state) or
  `ValueError` (user input).
- Pydantic models use `model_config = ConfigDict(...)`, never the
  legacy `class Config:` block.
- Read CLAUDE.md before changing things that touch session state,
  the operator overloads, or OpenTelemetry context handling — it
  documents past regressions and the reasoning behind them.

## Commit hygiene

- One topic per commit; the commit message body should explain the
  *why*, not just the *what*.
- Update `CHANGELOG.md` (under `## [Unreleased]`) for user-visible
  changes.

## Reporting issues

GitHub Issues are the canonical channel:
<https://github.com/juchengquan/llmagpie/issues>.
