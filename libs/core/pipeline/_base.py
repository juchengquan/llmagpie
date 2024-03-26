from abc import abstractmethod
import weakref
from pydantic import BaseModel, ConfigDict,Field, model_validator
# typing
from typing import Sequence, Dict, List, Optional, Callable, Union, Any

from llmagpie.core.dag import SingleDAG
from llmagpie.core.node import BaseNode
from llmagpie.core.node_disposable import BaseNodeDisposable

class BasePipelineMixin(BaseModel):
    model_config: ConfigDict = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True
    )
    complied: bool = False
    graph: SingleDAG = Field(default_factory=SingleDAG)
    
    # NOT_USED
    def _check_complie_status(self):
        assert self.compiled, "The pipeline is not complied."
    

    def __init__(self, nodes: List = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_nodes(nodes)
    
    def add_node(self, node, name: str=None):
        self._add_node(node, name) if name else self._add_node(node, node.name) 
        
    def add_nodes(self, nodes):
        if isinstance(nodes, Sequence):
            for n in nodes:
                self._add_node(n, n.name)
        elif isinstance(nodes, Dict):
            for n_name, n in nodes.items():
                self._add_node(n, n_name)
        
    def _add_node(self, node: BaseNode, name: str):
        if node.graph != self.graph:
            self.graph.update(node.graph)
            node.graph = self.graph
        if node._id not in self.graph.nodes:
            self.graph.add_node(node._id, name=node.name, _obj=weakref.ref(node))
    
    def add_edge(self, 
        src_node: Union[BaseNode, BaseNodeDisposable],
        dest_node: Union[BaseNode, BaseNodeDisposable],
        src_key: Union[List[str], str] = None,
        dest_key: Union[List[str], str] = None,
        conditional_func: Optional[Callable] = lambda *args, **kwargs: True
    ):
        if isinstance(src_node, BaseNodeDisposable) and isinstance(dest_node, BaseNodeDisposable):
            src_node._add_edges(dest_node, upstream=False, conditional_funcs=conditional_func)
        elif isinstance(src_node, BaseNode) and isinstance(dest_node, BaseNode):
            BaseNodeDisposable(node=src_node, keys=src_key)._add_edges(
                BaseNodeDisposable(node=dest_node, keys=dest_key),
                upstream=False,
                conditional_funcs=conditional_func
            )
        else:
            raise TypeError
        
    @abstractmethod
    def compile(self):
        """"""
        # weakref
    