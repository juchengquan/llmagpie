from __future__ import annotations  # self reference

from pydantic import BaseModel, Field
# typing
from typing import Dict, Optional, Union, Callable, List, Any


class BaseNodeDisposable(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        
    node: Any  # TODO: from llmagpie.core.node.base_node import BaseNode
    keys: List[str]
    keys_internal: Dict = Field(default_factory=dict)
    conditional_func: Optional[Callable] = None
    
    def _post_init(self):
        self.keys_internal = {
            "as_input": {self.node._io_key_mapping["input"].get(k) for k in self.keys},
            "as_output": {self.node._io_key_mapping["output"].get(k) for k in self.keys},
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._post_init()
        
    def __or__(self, conditional_func: Callable):
        if not isinstance(conditional_func, Callable):
            raise TypeError("Input must be a function")
        self.conditional_func = conditional_func
        return self
    
    def __lshift__(self, node_disposables: "BaseNodeDisposable") -> "BaseNodeDisposable":
        self._add_edges(node_disposables, upstream=True)
        return node_disposables
        
    def __rshift__(self, node_disposables: "BaseNodeDisposable") -> "BaseNodeDisposable":
        self._add_edges(node_disposables, upstream=False)
        return node_disposables
    
    def __rlshift__(self, node_disposables):
        """Implement [X] >> X because list does not have __rshift__ operators.
        """
        self.__rshift__(node_disposables)
        return self
    
    def __rrshift__(self, node_disposables):
        """Implement [X] << X because list does not have __lshift__ operators.
        """
        self.__lshift__(node_disposables)
        return self
    
    def _add_edges(self, 
        node_disposables: Union[List["BaseNodeDisposable"], "BaseNodeDisposable"], 
        upstream: bool = False,
        conditional_funcs: Optional[Union[Callable, List[Callable]]] = None,
    ):
        if not isinstance(node_disposables, List):
            node_disposables = [node_disposables]
        if conditional_funcs is None:
            conditional_funcs = [lambda *args, **kwargs: True] * len(node_disposables)
        if isinstance(conditional_funcs, Callable):
            conditional_funcs = [conditional_funcs]
        
        if len(conditional_funcs) != len(node_disposables):
            raise ValueError("The length of conditional functions and nodes must be same.")
        
        for _node, _conditional_func in zip(node_disposables, conditional_funcs):
            if upstream:
                upper, lower = _node, self
            else:
                upper, lower = self, _node
            
            if lower.conditional_func:
                _conditional_func = lower.conditional_func
            
            # TODO check keys
            _conditional_func = convert_func(_conditional_func, upper.node._io_key_mapping["output_internal"])
            
            for key in lower.keys_internal["as_input"]:
                lower.node._input_keys_nodes_map[key] = lower.node._input_keys_nodes_map.get(key, [])
                lower.node._input_keys_nodes_map[key].append(upper.node._id)
            
            lower.node._input_keys_binded.update(lower.keys_internal["as_input"])
            
            lower.node.is_start_node = False
            upper.node.is_end_node = False
            
            # update graph TODO
            if upper.node.graph != lower.node.graph:
                lower.node.graph.update(upper.node.graph)
                del upper.node.graph
                upper.node.graph = lower.node.graph
            # update edge
            upper.node.graph.add_edge(
                u_of_edge=upper.node._id,
                v_of_edge=lower.node._id,
                #
                _output_keys=upper.keys_internal["as_output"],
                _input_keys=lower.keys_internal["as_input"],
                _conditional_func=_conditional_func,
            )


from functools import wraps, partial
def convert_func(func: Optional[Callable]=None, mapping: Dict={}):
    if func is None:
        return partial(convert_func, mapping=mapping)
    
    @wraps(func)
    def _wrapper(*args, **kwargs):
        _kwargs = {mapping[k]: v for k, v in kwargs.items()}
        return func(*args, **_kwargs)
    
    return _wrapper
          
     