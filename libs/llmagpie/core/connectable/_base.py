from __future__ import annotations

import uuid
from asyncio import get_running_loop, new_event_loop
from logging import Logger

from pydantic import Field, BaseModel, PrivateAttr
from pydantic._internal._model_construction import ModelMetaclass

from deprecated import deprecated
from llmagpie.core.logging import get_or_create_logger
# typing
from typing import List, Dict, Union, Set, Literal, Optional, Generator, Callable, AsyncGenerator, cast
from abc import abstractmethod
from enum import Enum
from llmagpie.core.state import BaseState, InternalDictState

from llmagpie.core.utilities.async_to_sync import (
    exec_generator_in_event_loop, exec_generator_in_separated_thread
)

class _MetaFoo(ModelMetaclass):
    @property
    def class_name(self) -> str:
        """class_name"""
        return self.__class__.__name__
   
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _RunningStatus(Enum):
    INACTIVE = 0
    RUNNING = 1
    ERROR = 2

class BaseStateStore(BaseModel):
    """This class stores the state with upstream and downstream contexts.
    """
    class Config:
        extra = "forbid"
        arbitrary_types_allowed: bool = True
        
    input_state: InternalDictState = Field(default_factory=InternalDictState, description="Input object store that saves the input history of pipelines and nodes.")
    output_history_state: InternalDictState = Field(default_factory=InternalDictState, description="Output object store that saves the output history of all execution of nodes.")
    output_state: InternalDictState = Field(default_factory=InternalDictState, description="Output object store that saves the output history of all execution of nodes.")
    
    def clean_user_defined_states(self):
        for name, field in self.model_fields.items():
            obj = getattr(self, name)
            if isinstance(obj, BaseState):
                obj.clear()
                
class BaseConnectDisposable(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    connectable: BaseConnectable
    in_keys: List[str] = []
    out_keys: List[str] = []

    logger: Logger

    def __init__(self, *args, **kwargs):
        logger = get_or_create_logger(self.__class__.__name__)
        super().__init__(logger=logger, *args, **kwargs)

    def __lshift__(self, connect_disposable: "BaseConnectDisposable"):
        self.connectable.pipeline.add_edge(
            src_connectable=connect_disposable.connectable,
            dest_connectable=self.connectable,
            src_key=connect_disposable.out_keys,
            dest_key=self.in_keys,
        )
        return connect_disposable

    def __rshift__(self, connect_disposable: "BaseConnectDisposable"):
        self.connectable.pipeline.add_edge(
            src_connectable=self.connectable,
            dest_connectable=connect_disposable.connectable, 
            src_key=self.out_keys,
            dest_key=connect_disposable.in_keys,  
        )
        
        return connect_disposable

    def __rrshift__(self, connect_disposable: "BaseConnectDisposable"):
        """Implement [BaseRunnable] >> BaseRunnable because list don't have __rshift__ operators.
        Note that self refer to BaseConnectDisposable.
        """
        self.__lshift__(connect_disposable)
        return self

    def __rlshift__(self, connect_disposable: "BaseConnectDisposable"):
        """Implement [BaseConnectDisposable] << BaseRunnable because list don't have __lshift__ operators.
        Note that self refer to BaseConnectDisposable.
        """
        self.__rshift__(connect_disposable)
        return self


class BaseConnectable(BaseStateStore): # , metaclass=_MetaFoo):
    """This is base connectable including node and pipeline"""  # TODO
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
    
    _id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)
    name: str = Field()
    connectable_type: Literal["Pipeline", "Tool", "BaseNode"]
    logger: Logger = Field(default_factory=lambda: get_or_create_logger(logger_name="default"))
    is_start: bool = Field(default=True, description="Indicator that if the component is a start node.")
    is_end: bool = Field(default=True, description="Indicator that if the component is an end node.")
    _running_status: _RunningStatus = PrivateAttr(default=_RunningStatus.INACTIVE)
    
    pipeline: Optional[BaseConnectable] = Field(default=None)  # TODO: This is prepration for all connectables
    # TODO: typing might be wrong
        
    _input_keys_binded: Set[str] = set()
    # input binded keys
    _input_keys_nodes_map: Dict[str, List[str]] = {}
    # input keys of nodes map

    # PRIVATE
    func_schema: FunctionSchema = Field(default_factory=FunctionSchema)
    # condition function
    cond_func: Optional[Callable] = None
    inputs_to_cond: Optional[Dict] = None
    # TODO LOOP
    iteration_counter: Dict = Field(default_factory=dict)
    max_iteration_limit: int = Field(default=10)
    count_visited: int = 0
    _max_count_visited = 10

    def __init__(self, *args, **kwargs):
        kwargs["logger"] = get_or_create_logger(self.__class__.__name__)
        super().__init__(*args, **kwargs)

    def __lshift__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        return BaseConnectDisposable(connectable=self, in_keys=keys)

    def __rshift__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        return BaseConnectDisposable(connectable=self, out_keys=keys)

    def __rrshift__(self, keys: Union[str, List[str]]):
        return self.__lshift__(keys)

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
    async def async_event_on_execution(self,
        inputs: Optional[Dict],
        session_id: str,
        **kwargs
    ) -> AsyncGenerator:
        raise NotImplementedError

    def _error_callback(self, session_id: str, exc: Union[Exception, BaseException]):
        self.clean_states(session_id)
        self._running_status = _RunningStatus.ERROR
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
            
            self.clean_user_defined_states()
            # clean self
            for object_store_name in ["input_state", "output_state", "output_history_state"]:
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
                self.logger.warning(
                    f'{self.__class__.__name__}:{self.name}: '
                    f'Input pamatemeters {set(inputs.keys())} does not align with the keys: {set(self.func_schema.internal.input.all)}'
                    f', or Required input parameters {set(self.func_schema.internal.input.required)} '
                    f'does not align with the input keys: {set(inputs.keys())}')
                
                self.logger.warning(f"[PRECHECK] NOT EXECUTED YET -> {self.name}: Imcomplete inputs")  # TODO: raise error
                return None
            return inputs
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
    
    def invoke(
        self,
        inputs: Dict,
        session_id: Optional[str] = None,
        **kwargs
    )-> Generator:
        try:
            _thread_mode = False
            _is_new_loop = False
            aioloop = get_running_loop() # if there is not, go to RuntimeError
            if aioloop and aioloop.is_running():
                _thread_mode = True
                raise RuntimeError
        except RuntimeError:
            _is_new_loop = True
            aioloop = new_event_loop()
                
        session_id = uuid.uuid4().hex if not session_id else session_id
        try:
            if self.connectable_type == "Pipeline":
                assert getattr(self, "is_compiled", None), f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."    
            _inputs = self.precheck(session_id, inputs)
            
            async_result = cast(AsyncGenerator,
                self.async_event_on_execution(
                    inputs=_inputs,
                    session_id=session_id,
                    **kwargs
                )
            )
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)

        try:
            if _thread_mode:
                yield from exec_generator_in_separated_thread(async_generator=async_result, loop=aioloop)
            else:
                yield from exec_generator_in_event_loop(async_generator=async_result, loop=aioloop)
                        
            if _is_new_loop:
                aioloop.close()
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
            
            async_result = cast(AsyncGenerator,
                self.async_event_on_execution(
                    inputs=_inputs,
                    session_id=session_id,
                    **kwargs
                )
            )
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
            
        try:
            return async_result
        except (BaseException, Exception) as exc:
            self._error_callback(session_id, exc)
        finally:
            self.clean_states(session_id)

    def _get_from_local_store(self, session_id: str) -> Dict:
        dt_local_store = {}

        for _key, _node_id_list in self._input_keys_nodes_map.items():
            all_inputs_on_key = []
            for input_dict in self.input_state.get(session_id, {}).get(_key, []):
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
