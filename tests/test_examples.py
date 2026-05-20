"""Smoke-test every example under `_examples/simple_composition/` by running
it as a subprocess and asserting it exits cleanly. The examples exercise the
full node/pipeline/DAG machinery and act as integration tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "_examples" / "simple_composition"


def _discover_examples():
    return sorted(p for p in _EXAMPLES_DIR.glob("*.py") if p.name != "__init__.py")


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
    env["PYTHONPATH"] = str(_REPO_ROOT / "libs") + os.pathsep + env.get("PYTHONPATH", "")

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
