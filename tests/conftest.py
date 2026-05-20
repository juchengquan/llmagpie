import sys
from pathlib import Path

# Make the `libs/` layout importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "libs"))
