from types import MethodType
from pydantic._internal._model_construction import ModelMetaclass
from ._base import BasePipelineMixin


class MultiHeadPipeline(BasePipelineMixin):
    def _validate(self):
        self._validate_root_nodes()

    def _validate_root_nodes(self):
        """"""
        assert len(self.graph.head_nodes) >= 1, f"Only one root is not allowed in {self.__class__.__name__}!"
