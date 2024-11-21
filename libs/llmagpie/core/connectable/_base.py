from __future__ import annotations
# import os
import uuid
import itertools
import asyncio
from logging import Logger

from pydantic import Field, BaseModel, PrivateAttr, computed_field
from pydantic._internal._model_construction import ModelMetaclass

from llmagpie.core.logging import get_or_create_logger
# typing
from llmagpie.core.nodes.disposable import BaseNodeDisposable
from typing import List, Dict, Union, Any, Awaitable, Set, Optional, Generator, Callable, AsyncGenerator, cast
from abc import abstractmethod
from deprecated import deprecated

class _MetaFoo(ModelMetaclass):
    @property
    def class_name(self) -> str:
        """class_name"""
        return self.__class__.__name__

class BaseStateStore(BaseModel):
    """This class stores the state with upstream and downstream contexts.
    """
    input_state: Dict[str, Dict[str, Dict]] = Field(default_factory=dict, description="Input object store that saves the lastest info from parent nodes (multiple sources are allowed).")
    history_state: Dict[str, Dict[str, List[Dict]]] = Field(default_factory=dict, description="Input object store that saves the input history of pipelines and nodes.")
    output_state: Dict[str, List[Dict]] = Field(default_factory=dict, description="Output object store that saves the output history of all execution of nodes.")
      
# Function Schema DataClass
class _InterChangableInferface(BaseModel):
    required: List = []
    all: Dict = {}
    
class _FunctionSchema(BaseModel):
    input: _InterChangableInferface = Field(default_factory=_InterChangableInferface)
    output: _InterChangableInferface = Field(default_factory=_InterChangableInferface)
    
class FunctionSchema(BaseModel):
    internal: _FunctionSchema = Field(default_factory=_FunctionSchema)
    external: _FunctionSchema = Field(default_factory=_FunctionSchema)


class BaseConnectable(BaseStateStore): # , metaclass=_MetaFoo):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
    
    _id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)
    name: str = Field()
    connectable_type: str = Field()
    logger: Logger = Field(default_factory=lambda: get_or_create_logger(logger_name="default"))
    is_start: bool = Field(default=True, description="Indicator that if the component is a start node.")
    is_end: bool = Field(default=True, description="Indicator that if the component is an end node.")

    _input_keys_binded: Set[str] = set()
    # input binded keys
    _input_keys_nodes_map: Dict[str, str] = {}
    # input keys of nodes map
    
    pipeline: Optional[Any] = None  # TODO: This is prepration for all connectables

    # PRIVATE
    func_schema: FunctionSchema = Field(default_factory=FunctionSchema)
    
    # condition function
    cond_func: Optional[Callable] = None
    inputs_to_cond: Optional[Dict] = None
    # LOOP
    iteration_counter: Dict = Field(default_factory=dict)
    max_iteration_limit: int = Field(default=10)
    # TODO cqju
    count_visited: int = 0
    _max_count_visited = 10

    def __init__(self, *args, **kwargs):
        kwargs["logger"] = get_or_create_logger(self.__class__.__name__)
        super().__init__(*args, **kwargs)

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
    async def event_on_execution(self,
        inputs: Optional[Dict],
        session_id: str,
        **kwargs
    ) -> AsyncGenerator:
        raise NotImplementedError

    def _error_callback(self, session_id: str, exc: Union[Exception, BaseException]):
        self.clean_states(session_id)
        self.logger.error(f"Error on {self.name} -> {exc}")
        raise exc

    def clean_states(self, session_id: str):
        """"""
        try:
            if hasattr(self, "graph"):
                graph = getattr(self, "graph")
                for _id in graph.nodes:
                    _node = graph.nodes[_id]["_obj"]
                    _node.clean_states(session_id)
                
            # clean self
            for object_store_name in ["input_state", "history_state", "output_state"]:
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
        inputs: Optional[Dict] = None,
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
                set(inputs.keys()).issubset(self.func_schema.internal.input.all) and \
                    set(self.func_schema.internal.input.required).issubset(set(inputs.keys()))
            ):  
                self.logger.warning(f'{self.__class__.__name__}:{self.name}: Input pamatemeters {set(inputs.keys())} does not align with the keys: {set(self.func_schema.internal.input.all)} \
                    or Required input parameters {set(self.func_schema.internal.input.required)} does not align with the input keys: {set(inputs.keys())}')
                
                self.logger.warning(f"[PRECHECK] NOT EXECUTED YET -> {self.name}")  # TODO: raise error
                return None

            if not self.is_start:
                if self.input_state[session_id] not in self.history_state.get(session_id, {}).values():
                    # TODO
                    for _key, _value in self.input_state[session_id].items():
                        self.history_state[session_id] = self.history_state.get(session_id, {})
                        self.history_state[session_id][_key] = \
                            self.history_state.get(session_id, {}).get(_key, []) + [_value]
            
            return inputs
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

    def invoke(
        self,
        inputs: Dict,
        session_id: Optional[str] = None,
        **kwargs
    )-> Union[Generator[Dict], Dict]:
        session_id = uuid.uuid4().hex if not session_id else session_id
        
        try:
            if self.connectable_type == "Pipeline":
                assert getattr(self, "is_compiled", None), f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."    
            _inputs = self.precheck(session_id, inputs)
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

        try:
            async_result = cast(AsyncGenerator,
                self.event_on_execution(
                    inputs=_inputs,
                    session_id=session_id,
                    **kwargs
                )
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
            self.clean_states(session_id)

    async def async_invoke(
        self,
        inputs: Dict,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator:
        session_id = uuid.uuid4().hex if not session_id else session_id
        
        try:
            if self.connectable_type == "Pipeline":
                assert getattr(self, "is_compiled"), f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."    
            _inputs = self.precheck(session_id, inputs)
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
        
        try:
            async_result = self.event_on_execution(
                inputs=_inputs,
                session_id=session_id,
                **kwargs
            )
            async for res in cast(AsyncGenerator, async_result):
                yield res
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
        finally:
            self.clean_states(session_id)

    def _get_from_local_store(self, session_id: str) -> Dict:
        dt_local_store = {}

        for _key, _node_id_list in self._input_keys_nodes_map.items():
            all_inputs_on_key = list(
                ele for ele in self.input_state[session_id][_key].values() if ele
            )
            for input_dict in self.history_state.get(session_id, {}).get(_key, []):
                all_inputs_on_key.extend(
                    list( ele for ele in input_dict.values() if ele )
                )

            all_inputs_on_key.sort(key=lambda x: -x["_timestamp"])

            for val in all_inputs_on_key:
                if val["value"] is not None:
                    dt_local_store.update({_key: val["value"]})
                    break
        return dt_local_store

    @deprecated
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
