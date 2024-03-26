from __future__ import annotations
from asyncio import create_task

import uuid
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, computed_field
from inspect import isawaitable

from .dag import DAG
from ._enum import NodeType
# typing
from .dag import GraphWithAPI
from typing import Sequence, Dict, Optional, Union, Tuple

class BaseNodeMixin(BaseModel):
    model_config: ConfigDict = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True
    )
    
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    
    @computed_field # TODO
    @property
    def name(self) -> str:
        return self.__class__.__name__ + "|" + self.id[-4:]

class NodeGroup(Dict[str, BaseNodeMixin]):
    """"""
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.__repr_str__(", ")})'

    def __repr_str__(self, join_str: str):
        return join_str.join(self.keys())
    
class Node(BaseNodeMixin):   
    node_type: NodeType
     
    prev_nodes: NodeGroup = Field(default=NodeGroup())
    next_nodes: NodeGroup = Field(default=NodeGroup())
    
    graph: GraphWithAPI = Field(default=DAG) # Optional[GraphWithAPI] = None # 
    
    local_object_store: Dict = Field(default=dict())
    # def has_dag(self) -> bool:
    #     return self.graph is not None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if not self.has_dag():
        #     self.graph = DAG
        # print(self.graph)
        
        self.graph.add_node(
            self.id,
            name=self.name,
            object=self
        )
        
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'

    def __lshift__(self, other_nodes: Union[Node, Sequence[Node]]) -> Node:
        self._add_adjacent_nodes(other_nodes, lshift=True)
        return other_nodes
        
    def __rshift__(self, other_nodes: Union[Node, Sequence[Node]]):
        self._add_adjacent_nodes(other_nodes, lshift=False)
        return other_nodes
    
    def __rlshift__(self, other_nodes):
        self.__rshift__(other_nodes)
        return self
    
    def __rrshift__(self, other_nodes):
        self.__lshift__(other_nodes)
        return self
    
    def _add_adjacent_nodes(self, other_nodes: Union[Node, Sequence[Node]], lshift: bool):
        if isinstance(other_nodes, Node):
            other_nodes: Sequence[Node] = [other_nodes]
            
        if lshift:
            for ele_node in other_nodes:
                ele_node.next_nodes.update({
                    self.id: self
                })
                self.prev_nodes.update({
                    ele_node.id: ele_node
                })
                self.graph.add_edge(ele_node.id, self.id)
        else:
            for ele_node in other_nodes:
                self.next_nodes.update({
                    ele_node.id: ele_node
                })
                ele_node.prev_nodes.update({
                    self.id: self
                })
                
                self.graph.add_edge(self.id, ele_node.id)
    
    def _check_adjacency(self):
        # if len(self.prev_nodes) != self.num
        """"""
        
    async def __call__(self, session_id: str):
        # checking 
        self._check_adjacency()
        # execution
        _curr_outputs = self.execute(session_id)
        if isawaitable(_curr_outputs):
            _curr_outputs = await _curr_outputs
        
        if isinstance(_curr_outputs, Tuple):
            _curr_outputs, _conditional_list = _curr_outputs
        else:
            _conditional_list = None
            
        # release 
        await self._init_current_node_values(session_id)
        await self.emit(output=_curr_outputs, session_id=session_id, conditional_list=_conditional_list)
        
        return _curr_outputs
    
    async def emit(self, output, *args, session_id: str = None, conditional_list: Dict[str, bool] = None, **kwargs):
        if conditional_list is None or conditional_list == {}:
            conditional_list = {
                ele_node.id: True for ele_node in self.next_nodes.values()
            }
        
        if len(conditional_list) != len(self.next_nodes):
            raise Exception

        # emission
        for _, ele_node in self.next_nodes.items():
            if conditional_list[ele_node.id]:
                # to children
                ele_node.local_obect_store[session_id] = ele_node.local_object_store.get(session_id, {})
                ele_node.local_object_store[session_id][self.id] = output
                
                create_task(ele_node._event_on_execution(session_id))
                
    async def _event_on_execution(self, session_id: str):
        local_store = [self.local_object_store[session_id].get(ele_node.id, None) for ele_node in self.prev_nodes.values()]
        if all(local_store):
            await self.__call__(session_id=session_id)
    
    async def _init_current_node_values(self, session_id):
        self.local_object_store.pop(session_id)
        
    @abstractmethod
    async def execute(self, session_id: str, *args, **kwargs):
        raise NotImplementedError
    