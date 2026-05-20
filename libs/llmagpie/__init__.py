from importlib import metadata as _metadata

from . import base, core, experimental
from .base import BaseNode, BasePipeline, MakeNode

try:
    __version__ = _metadata.version(__package__)  # type: ignore
except _metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = ["BaseNode", "BasePipeline", "MakeNode", "base", "core", "experimental"]
