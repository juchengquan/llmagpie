"""Smoke-test every example under `_examples/` by running it as a
subprocess and asserting it exits cleanly. The examples exercise the
full node/pipeline/DAG machinery (in `simple_composition/`) and the
high-level Agent abstraction (in `agents/`), so they double as
integration tests for the framework."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_ROOT = _REPO_ROOT / "_examples"


def _discover_examples():
    paths: list[Path] = []
    for example_dir in sorted(_EXAMPLES_ROOT.iterdir()):
        if not example_dir.is_dir():
            continue
        for f in sorted(example_dir.glob("*.py")):
            if f.name.startswith("_"):  # helper modules (e.g. _mock.py) aren't entry points
                continue
            paths.append(f)
    return paths


def _coverage_active() -> bool:
    """Detect whether the parent pytest is running with coverage enabled.

    pytest-cov sets COV_CORE_SOURCE when collecting; we also fall back to
    checking whether `coverage` itself thinks tracing is active.
    """
    if any(k in os.environ for k in ("COV_CORE_SOURCE", "COV_CORE_CONFIG", "COVERAGE_RUN")):
        return True
    try:
        import coverage

        return coverage.Coverage.current() is not None
    except Exception:
        return False


@pytest.mark.parametrize("example_path", _discover_examples(), ids=lambda p: p.stem)
def test_example_runs_cleanly(example_path: Path):
    env = os.environ.copy()
    # libs/ for `from llmagpie import ...`; the example's own dir for
    # sibling helpers like `_examples/agents/_mock.py` (`from _mock import ...`).
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_REPO_ROOT / "libs"),
            str(example_path.parent),
            env.get("PYTHONPATH", ""),
        ]
    )

    if _coverage_active():
        # Wrap the subprocess in `coverage run` so its instrumentation merges
        # into the parent pytest-cov report (parallel-mode names files
        # uniquely; pytest-cov combines them at session end because
        # `[tool.coverage.run] parallel = true`).
        cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--rcfile={_REPO_ROOT / 'pyproject.toml'}",
            "--parallel-mode",
            str(example_path),
        ]
    else:
        cmd = [sys.executable, str(example_path)]

    # cwd stays at the repo root so the `[tool.coverage.run] source`
    # relative path resolves correctly; sibling helper imports are
    # handled via PYTHONPATH (above).
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Example {example_path.name} exited with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
