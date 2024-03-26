# from abc import abstractmethod
import uuid
from pydantic import BaseModel, ConfigDict,Field
# typing
from typing import Sequence, Dict, List, Optional, Callable, Union, Any
from types import MethodType

from llmagpie.core.dag import DAG
from llmagpie.core.node import BaseNode
from llmagpie.core.node_disposable import BaseNodeDisposable
from llmagpie.core.function import fire_single

from ._base import BasePipelineMixin

class SinglePipeline(BasePipelineMixin):
    def validate(self):
        """"""
        self.graph.validate()
        return self
    
    def compile(self):
        self._make_fire()
        self.validate()
        
    def _make_fire(self):
        """"""
        object.__setattr__(self, "fire", MethodType(fire_single, self))
        