from __future__ import annotations
# import os
import uuid
import itertools
from logging import Logger

from pydantic import Field, BaseModel, PrivateAttr, computed_field

from llmagpie.core.dag import SingleDAG
from llmagpie.core.logging import get_or_create_logger
# typing
from llmagpie.core.nodes.disposable import BaseNodeDisposable
from typing import List, Dict, Union, Any, Awaitable, Set, Optional
from abc import abstractmethod


class BaseConnectable(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    logger: Logger = None
    name: str = Field()
    node_type: str

    is_start: bool = Field(default=True, description="Indicator that if the component is a start node.")
    is_end: bool = Field(default=True, description="Indicator that if the component is an end node.")

    pipeline: Optional[Any] = None  # TODO: This is prepration for connectable

    _id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)
    # graph: Optional[SingleDAG] = None

    input_object_store: Dict = Field(default_factory=dict, description="Input object store that saves the lastest info from parent nodes (multiple sources are allowed).")
    stack_object_store: Dict = {}  # TODO LC: Add description
    output_object_store: Dict = {}  # TODO LC: Add description
    
    # PRIVATE
    _input_keys_binded: Set[str] = set()
    _input_keys_nodes_map: Dict[str, str] = {}
    
    _input_schema_all: Dict = PrivateAttr(default_factory=dict)
    _input_schema_required: Dict = PrivateAttr(default_factory=dict)
    _output_schema_all: Dict = PrivateAttr(default_factory=dict)

    # LOOP
    iteration_counter: Dict = Field(default_factory=dict)
    max_iteration_limit: int = Field(default=10)
    # TODO cqju
    count_visited: int = 0
    _max_count_visited = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_or_create_logger(self.__class__.__name__)

    def __lshift__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        return BaseNodeDisposable(connectable=self, in_keys=keys)

    def __rrshift__(self, keys: Union[str, List[str]]):
        return self.__lshift__(keys)

    def __rshift__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        return BaseNodeDisposable(connectable=self, out_keys=keys)

    def __rlshift__(self, keys: Union[str, List[str]]):
        return self.__rshift__(keys)

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return self.__repr__()

    @computed_field
    @property
    def class_name(self) -> str:
        """class_name"""
        return self.__class__.__name__

    # @computed_field
    # @property
    # def obj_name(self) -> str:
    #     """obj name"""
    #     return self.__class__.__name__ + "|" + self._id[-4:]

    @abstractmethod
    def _validate(self):
        raise NotImplementedError
    
    @abstractmethod
    def precheck(self):
        raise NotImplementedError
    
    @abstractmethod
    async def event_on_execution(self):
        raise NotImplementedError

    def clean_object_store(self, session_id: str):
        """"""
        try:
            if hasattr(self, "graph"):
                for _id in self.graph.nodes:
                    _node = self.graph.nodes[_id]["_obj"]
                    _node.clean_object_store(session_id)
                
            # clean self
            for object_store_name in ["input_object_store", "stack_object_store", "output_object_store", "iteration_counter"]:
                if hasattr(self, object_store_name):
                    self.logger.debug(f"{self.name} - {object_store_name}: BEFORE")
                    self.logger.debug(getattr(self, object_store_name))
                    getattr(self, object_store_name).pop(session_id, None)
                    self.logger.debug(f"{self.name} - {object_store_name}: AFTER")
                    self.logger.debug(getattr(self, object_store_name))
        except Exception as exc:
            raise exc

    
    def _get_local_store(self, session_id: str) -> Dict:
        dt_local_store = {}
        for _key, _node_id_list in self._input_keys_nodes_map.items():
            # TODO LC: need to double check the logic here
            all_inputs_on_key = list(itertools.chain.from_iterable([
            self.input_object_store.get(session_id).get(_key).get(_node_id, []) for _node_id in _node_id_list
            ]))
            all_inputs_on_key.sort(key=lambda x: -x["_timestamp"])
            for val in all_inputs_on_key:
                if val["value"] is not None:
                    dt_local_store.update({_key: val["value"]})
                    break  # TODO: to only fetch the first valid value if no condition is provided
            if not dt_local_store.get(_key): 
                ## input missing, check whether required input is missing.
                if _key in self._input_schema_required["internal"]:
                    self.logger.warning(f'Required input "{_key}" missing, attempt using previous value.')
                    all_inputs_on_key = list(itertools.chain.from_iterable([
                        self.stack_object_store.get(session_id).get(_key).get(_node_id, []) for _node_id in _node_id_list
                    ]))
                    all_inputs_on_key.sort(key=lambda x: -x["_timestamp"])
                for val in all_inputs_on_key:
                    if val["value"] is not None:
                        dt_local_store.update({_key: val["value"]})
                        break
        return dt_local_store

