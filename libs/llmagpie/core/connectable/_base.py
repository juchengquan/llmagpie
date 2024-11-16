from __future__ import annotations
# import os
import uuid
import itertools
import asyncio
from logging import Logger

from pydantic import Field, BaseModel, PrivateAttr, computed_field
from pydantic._internal._model_construction import ModelMetaclass

from llmagpie.core.dag import SingleDAG
from llmagpie.core.logging import get_or_create_logger
# typing
from llmagpie.core.nodes.disposable import BaseNodeDisposable
from typing import List, Dict, Union, Any, Awaitable, Set, Optional
from abc import abstractmethod


class _MetaFoo(ModelMetaclass):
    @property
    def class_name(self) -> str:
        """class_name"""
        return self.__class__.__name__

class BaseConnectable(BaseModel, metaclass=_MetaFoo):
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

    input_object_store: Dict[str, Dict[str, Dict]] = Field(default_factory=dict, description="Input object store that saves the lastest info from parent nodes (multiple sources are allowed).")
    history_object_store: Dict[str, Dict[str, List[Dict]]] = Field(default_factory=dict, description="Input object store that saves the input history of pipelines and nodes.")
    output_object_store: Dict[str, Dict[str, Dict]] = Field(default_factory=dict, description="Output object store that saves the output history of all execution of nodes.")
    # PRIVATE
    _input_keys_binded: Set[str] = set()
    _input_keys_nodes_map: Dict[str, str] = {}
    
    _input_schema_all: Dict = PrivateAttr(default_factory=dict)
    _input_schema_required: Dict = PrivateAttr(default_factory=dict)
    _output_schema_all: Dict = PrivateAttr(default_factory=dict)

    # TODO: condition function
    cond_func: Callable = None
    inputs_to_cond: Dict = None
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

    # @computed_field
    # @property
    # def obj_name(self) -> str:
    #     """obj name"""
    #     return self.__class__.__name__ + "|" + self._id[-4:]

    @abstractmethod
    def _validate(self):
        raise NotImplementedError

    @abstractmethod
    async def event_on_execution(self):
        raise NotImplementedError

    def _error_callback(self, session_id: str, exc: Union[Exception, BaseException]):
        self.clean_object_store(session_id)
        self.logger.error(f"Error on {self.name} -> {exc}")
        raise exc

    def clean_object_store(self, session_id: str):
        """"""
        try:
            if hasattr(self, "graph"):
                for _id in self.graph.nodes:
                    _node = self.graph.nodes[_id]["_obj"]
                    _node.clean_object_store(session_id)
                
            # clean self
            for object_store_name in ["input_object_store", "history_object_store", "output_object_store"]:
                if hasattr(self, object_store_name):
                    self.logger.debug(f"{self.name} - {object_store_name}: BEFORE")
                    self.logger.debug(getattr(self, object_store_name))
                    getattr(self, object_store_name).pop(session_id, None)
                    self.logger.debug(f"{self.name} - {object_store_name}: AFTER")
                    self.logger.debug(getattr(self, object_store_name))
        except Exception as exc:
            raise exc

    def precheck(
        self,
        session_id: str,
        inputs: Dict = None,
        **kwargs
    ) -> Union[Dict, None]:
        try:
            if inputs is None:
                inputs = self._get_from_local_store(session_id)

            if self.cond_func is not None:
                if self.inputs_to_cond:
                    inputs_to_cond_values = {k: inputs[v] for k, v in self.inputs_to_cond.items()}
                else:
                    self.logger.warning(f"{self} cond_func missing inputs_to_cond. Mapping all inputs of the node.")
                    inputs_to_cond_values = inputs
                if self.cond_func(**inputs_to_cond_values) == False:
                    self.logger.warning(f"[Condition not met] NOT EXECUTED -> {self.name}")
                    return None
                    
            if not (
                set(inputs.keys()).issubset(self._input_schema_all["internal"]) and set(self._input_schema_required["internal"]).issubset(set(inputs.keys()))
            ):  
                self.logger.warning(f'{self.__class__.__name__}:{self.name}: Input pamatemeters {set(inputs.keys())} does not align with the keys: {set(self._input_schema_all["internal"])} \
                    or Required input parameters {set(self._input_schema_required["internal"])} does not align with the input keys: {set(inputs.keys())}')
                
                self.logger.warning(f"[PRECHECK] NOT EXECUTED YET -> {self.name}")  # TODO: raise error
                return None

            if not self.is_start:
                if self.input_object_store[session_id] not in self.history_object_store.get(session_id, {}).values():
                    # TODO
                    for _key, _value in self.input_object_store[session_id].items():
                        self.history_object_store[session_id] = self.history_object_store.get(session_id, {})
                        self.history_object_store[session_id][_key] = \
                            self.history_object_store.get(session_id, {}).get(_key, []) + [_value]
            
            return inputs
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

    def invoke(
        self,
        inputs: Dict,
        session_id: str = None,
        **kwargs
    )-> Union[Generator[Dict], Dict]:
        try:
            if self.node_type == "Pipeline":
                assert self.is_compiled, f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."
            session_id = uuid.uuid4().hex if not session_id else session_id
            _inputs = self.precheck(session_id, inputs)
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

        try:
            async_result: AsyncGenerator = self.event_on_execution(
                inputs=_inputs,
                session_id=session_id,
                **kwargs
            )

            loop = asyncio.get_event_loop()
            while True:
                try:
                    yield loop.run_until_complete(async_result.__anext__())
                except StopAsyncIteration:
                    break
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
        finally:
            self.clean_object_store(session_id)

    async def async_invoke(
        self,
        inputs: Dict,
        session_id: str = None,
        **kwargs
    ) -> AsyncIterator:
        try:
            if self.node_type == "Pipeline":
                assert self.is_compiled, f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."
            session_id = uuid.uuid4().hex if not session_id else session_id
            _inputs = self.precheck(session_id, inputs)
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
        
        try:
            async_result = self.event_on_execution(
                inputs=_inputs,
                session_id=session_id,
                **kwargs
            )
            async for res in async_result:
                yield res
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
        finally:
            self.clean_object_store(session_id)


    def _get_from_local_store(self, session_id: str) -> Dict:
        dt_local_store = {}

        for _key, _node_id_list in self._input_keys_nodes_map.items():
            all_inputs_on_key = list(
                ele for ele in self.input_object_store[session_id][_key].values() if ele
            )
            for input_dict in self.history_object_store.get(session_id, {}).get(_key, []):
                all_inputs_on_key.extend(
                    list( ele for ele in input_dict.values() if ele )
                )

            all_inputs_on_key.sort(key=lambda x: -x["_timestamp"])

            for val in all_inputs_on_key:
                if val["value"] is not None:
                    dt_local_store.update({_key: val["value"]})
                    break
        return dt_local_store


    def _find_values(self, nested_dict: Dict, key: str, node_id_list: list) -> list:
        """Collect results from the current level.
        Args:
            nested_dict (dict): _description_
            key (str): _description_
            node_id_list (list): _description_

        Returns:
            _type_: _description_
        """
        results: List[Dict] = [
            nested_dict[key][node_id] for node_id in node_id_list if key in nested_dict.keys() and node_id in nested_dict[key]
        ]
        return results

    def _flatten_history_object_store(self, session_id) -> Dict[str, List[Dict]]:
        res = {
            session_id: {
                ".".join(_key.split('.')[1:]): _value for _key, _value in self.history_object_store.get(session_id, {}).items()
            }
        }
        return res