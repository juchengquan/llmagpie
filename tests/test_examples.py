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


@pytest.mark.parametrize("example_path", _discover_examples(), ids=lambda p: p.stem)
def test_example_runs_cleanly(example_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT / "libs") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Example {example_path.name} exited with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
