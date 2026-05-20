from importlib import metadata as _metadata

from . import base, core, experimental
from .base import BaseNode, BasePipeline, MakeNode
from .base.connectable import BaseConnectable

try:
    __version__ = _metadata.version(__package__)
except _metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [
    "BaseConnectable",
    "BaseNode",
    "BasePipeline",
    "MakeNode",
    "base",
    "core",
    "experimental",
]
