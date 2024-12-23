from importlib import metadata

try:
    __version__ = metadata.version(__package__)  # type: ignore
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

del metadata  # optional, avoids polluting the results of dir(__package__)


from . import base, core, experimental

from .base import BaseNode, MakeNode, BasePipeline

__all__ = [
    "BaseNode", "MakeNode", "BasePipeline",
    "base", "core", "experimental"
]