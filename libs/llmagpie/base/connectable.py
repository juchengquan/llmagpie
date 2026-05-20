from __future__ import annotations

import uuid
from abc import abstractmethod, ABC
from asyncio import get_running_loop, new_event_loop
from logging import Logger


from pydantic import ConfigDict, Field, BaseModel, PrivateAttr

from llmagpie.base.enum import NodeRunningStatus, ConnectableType
from llmagpie.base.logging import get_or_create_logger
from llmagpie.base.utils.async_to_sync import (
    exec_generator_in_event_loop, exec_generator_in_separated_thread
)
# typing
from typing import List, Dict, Union, Set, Literal, Optional, Generator, Callable, AsyncGenerator, cast, final



class BaseState:
    @abstractmethod    
    def clear(self):
        raise NotImplementedError
    
class ListState(List, BaseState):
    ...

class DictState(Dict, BaseState):
    ...

class InternalDictState(Dict, BaseState):
    ...


# Function Schema DataClass
class _InterchangeableInterface(BaseModel):
    required: List = Field(default_factory=list)
    all: Dict = Field(default_factory=dict)

class _FunctionSchema(BaseModel):
    input: _InterchangeableInterface = Field(default_factory=_InterchangeableInterface)
    output: _InterchangeableInterface = Field(default_factory=_InterchangeableInterface)

class FunctionSchema(BaseModel):
    internal: _FunctionSchema = Field(default_factory=_FunctionSchema)
    external: _FunctionSchema = Field(default_factory=_FunctionSchema)


class BaseStateStore(BaseModel):
    """This class stores the state with upstream and downstream contexts.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    input_state: InternalDictState = Field(default_factory=InternalDictState, description="Input object store that saves the input history of pipelines and nodes.")
    output_history_state: InternalDictState = Field(default_factory=InternalDictState, description="Output object store that saves the output history of all execution of nodes.")
    output_state: InternalDictState = Field(default_factory=InternalDictState, description="Output object store that saves the output history of all execution of nodes.")
    
    def clean_user_defined_states(self):
        # Access via the class (not the instance) to avoid the
        # PydanticDeprecatedSince211 warning on `self.model_fields`.
        for name in type(self).model_fields:
            obj = getattr(self, name)
            if isinstance(obj, BaseState):
                obj.clear()
                
class BaseConnectDisposable(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    connectable: BaseConnectable
    in_keys: List[str] = Field(default_factory=list)
    out_keys: List[str] = Field(default_factory=list)

    logger: Logger

    def __init__(self, *args, **kwargs):
        logger = kwargs.pop("logger", None) or get_or_create_logger(self.__class__.__name__)
        super().__init__(logger=logger, *args, **kwargs)

    def __lshift__(self, connect_disposable: "BaseConnectDisposable"):
        self.connectable.pipeline.add_edge(  # type: ignore
            src_connectable=connect_disposable.connectable,
            dest_connectable=self.connectable,
            src_key=connect_disposable.out_keys,
            dest_key=self.in_keys,
        )
        return connect_disposable

    def __rshift__(self, connect_disposable: "BaseConnectDisposable"):
        self.connectable.pipeline.add_edge(  # type: ignore
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


class BaseConnectable(BaseStateStore):
    """This is base connectable including node and pipeline"""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)
    name: str = Field()
    """The unique name of the tool that clearly communicates its purpose."""
    description: str = Field(default="")
    """Used to tell the model how/when/why to use."""
    connectable_type: ConnectableType
    logger: Logger = Field(default_factory=lambda: get_or_create_logger(logger_name="default"))
    is_start: bool = Field(default=True, description="Indicator that if the component is a start node.")
    is_end: bool = Field(default=True, description="Indicator that if the component is an end node.")
    _running_status: NodeRunningStatus = PrivateAttr(default=NodeRunningStatus.INACTIVE)
    
    # `pipeline` is a placeholder for all connectables
    # typing might be wrong 
    pipeline: Optional[BaseConnectable] = Field(default=None)
        
    # input bound keys
    _input_keys_bound: Set[str] = PrivateAttr(default_factory=set)
    # input keys of nodes map
    _input_keys_nodes_map: Dict[str, List[str]] = PrivateAttr(default_factory=dict)

    # PRIVATE
    func_schema: FunctionSchema = Field(default_factory=FunctionSchema)
    # Function Schema
    
    # condition function
    cond_func: Optional[Callable[..., bool]] = None
    inputs_to_cond: Optional[Dict[str, str]] = None
    # LOOP
    iteration_counter: Dict = Field(default_factory=dict)
    max_iteration_limit: int = Field(default=10)
    count_visited: int = 0

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("logger", get_or_create_logger(self.__class__.__name__))
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

    def _error_callback(self, session_id: str, exc: Exception):
        self.clean_states(session_id)
        self._running_status = NodeRunningStatus.ERROR
        self.logger.error(f"Error on {self.name} -> {exc}")
        raise exc

    def clean_states(self, session_id: str):
        """"""
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
                
                if not self.cond_func(**inputs_to_cond_values):
                    self.logger.warning(f"[Condition not met] NOT EXECUTED -> {self.name}")
                    return None
            
            if not (
                set(inputs.keys()).issubset(self.func_schema.internal.input.all) and \
                set(self.func_schema.internal.input.required).issubset(set(inputs.keys()))
            ):  
                self.logger.warning(
                    f'{self.__class__.__name__}:{self.name}: '
                    f'Input parameters {set(inputs.keys())} does not align with the keys: {set(self.func_schema.internal.input.all)}'
                    f', or Required input parameters {set(self.func_schema.internal.input.required)} '
                    f'does not align with the input keys: {set(inputs.keys())}')
                
                self.logger.warning(f"[PRECHECK] NOT EXECUTED YET -> {self.name}: Incomplete inputs")
                return None
            
            return inputs
        except Exception as exc:
            self._error_callback(session_id, exc)
    
    @final
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
            if self.connectable_type == ConnectableType.PIPELINE:
                assert getattr(self, "is_compiled", None), f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."    
            _inputs = self.precheck(session_id, inputs)
            
            async_result = cast(AsyncGenerator,
                self.async_event_on_execution(
                    inputs=_inputs,
                    session_id=session_id,
                    **kwargs
                )
            )
        except Exception as exc:
            self._error_callback(session_id, exc)

        try:
            if _thread_mode:
                yield from exec_generator_in_separated_thread(async_generator=async_result, loop=aioloop)
            else:
                yield from exec_generator_in_event_loop(async_generator=async_result, loop=aioloop)
                        
            if _is_new_loop:
                aioloop.close()
        except Exception as exc:
            self._error_callback(session_id, exc)
        finally:
            self.clean_states(session_id)
    
    @final
    async def async_invoke(
        self,
        inputs: Dict,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator:
        """Awaitable that returns an async generator. The returned generator
        owns the per-session cleanup: state for `session_id` is removed when
        iteration finishes (either StopAsyncIteration or exception)."""
        session_id = uuid.uuid4().hex if not session_id else session_id
        try:
            if self.connectable_type == ConnectableType.PIPELINE:
                assert getattr(self, "is_compiled"), f"Pipeline {self.name} is not compiled yet; please compile it first using `pipe.compile()`."
            _inputs = self.precheck(session_id, inputs)
            inner = cast(
                AsyncGenerator,
                self.async_event_on_execution(
                    inputs=_inputs,
                    session_id=session_id,
                    **kwargs,
                ),
            )
        except Exception as exc:
            # `_error_callback` raises; `clean_states` runs inside it.
            self._error_callback(session_id, exc)

        async def _wrapped() -> AsyncGenerator:
            try:
                async for state in inner:
                    yield state
            finally:
                self.clean_states(session_id)

        return _wrapped()

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
