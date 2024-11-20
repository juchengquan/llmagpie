# from types import MethodType
# from pydantic._internal._model_construction import ModelMetaclass
from ._base import BasePipelineMixin


class MultiHeadPipeline(BasePipelineMixin):
    def _validate(self):
        self._validate_root_nodes()
        # migrate heads and tails check from node level
        # Check in-degree and out-degree of nodes (on pipeline)
        for _id in self.graph.nodes:
            _node = self.graph.nodes[_id]["_obj"]
            assert (self.graph.in_degree(_id) == 0 and _node.is_start is True) \
                or (self.graph.in_degree(_id) != 0 and _node.is_start is not True), \
                f"{self.__class__.__name__} Node In-degree is wrong."
            assert (self.graph.out_degree(_id) == 0 and _node.is_end is True) \
                or (self.graph.out_degree(_id) != 0 and _node.is_end is not True), \
                f"{self.__class__.__name__} Node Out-degree is wrong."
        
    def _validate_root_nodes(self):
        """"""
        assert len(self.graph.head_nodes) >= 1, f"Only one root is not allowed in {self.__class__.__name__}!"
