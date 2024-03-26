from __future__ import annotations
from asyncio import create_task, CancelledError
import asyncio
import warnings
import uuid
import weakref

from abc import abstractmethod
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, computed_field, PrivateAttr, create_model,model_validator
from pydantic._internal._model_construction import ModelMetaclass
from inspect import isawaitable
from types import MethodType

from llmagpie.core.dag import DAG
from llmagpie.core.node_disposable import BaseNodeDisposable
from llmagpie.core.function import func_input_validator, create_schema_from_function
from llmagpie.core.function import fire_single
# typing
from typing import Awaitable, Sequence, Dict, Optional, Union, Tuple, List, Set,Any,Callable, Type


class BaseNodeMixin(BaseModel):
    __slots__ = ["__weakref__"]
    model_config: ConfigDict = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True
    )
    
    @computed_field
    @property
    def class_name(self) -> str:
        return self.__class__.__name__
    
    @computed_field # TODO
    @property
    def obj_name(self) -> str:
        return self.class_name + "|" + self._id[-4:]
    
    _id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)
    

class BaseNode(BaseNodeMixin):
    name: str = Field(default=None)
    graph: DAG = Field(default=DAG(name=uuid.uuid4().hex))
    
    input_keys: List[str] = []
    output_keys: List[str] = []
    
    @property
    def input_keys_internal(self):
        return list(self.NodeDataInput.schema()["properties"].keys())
    
    output_keys_internal: List[str]
    # FIXME
    NodeDataInput: ModelMetaclass
    NodeDataOutput: BaseModel = Field(default=None)
    
    is_start_node: bool = True
    is_end_node: bool = True
    
    ### PRIVATE
    _input_keys_nodes_map: Dict[str, str] = {}
    _downstream_nodes: List[str] = []
    
    input_object_store: Dict = {}
    output_object_store: Dict = {}
    
    _input_keys_binded: Set[str] = set()
    
    _io_key_mapping = Dict[str, Dict[str, str]]
    
    @model_validator(mode="after")
    def validate_io_keys(self):
        # TODO
        # wrap function
        # print(self.NodeDataInput)
        # self.NodeDataInput = create_schema_from_function(self.call)
        # print(self.NodeDataInput)
        
        if self.input_keys == []:
            self.input_keys = self.input_keys_internal
        if self.output_keys == []:
            self.output_keys = self.output_keys_internal
        
        return self
    
    # @classmethod
    # def create(cls, *args, **kwargs):
    #     output_keys_internal
    
    def __init__(self, *args, **kwargs):
        kwargs.update({
            "NodeDataInput": create_schema_from_function(self.call)
        })
        # print(create_schema_from_function(self.call))
        
        super().__init__(*args, **kwargs)
        
        if not self.name:
            self.name = self.obj_name
        
        _schema = {_n: (Any, Field()) for _n in self.output_keys_internal}
        self.NodeDataOutput = create_model(
            self.class_name + "Output",
            **_schema
        )
        
        # if not self.output_keys:
        #     self.output_keys = self.output_keys_internal

        # TODO
        # wrap function
        # self.NodeDataInput = create_schema_from_function(self.call)
        object.__setattr__(self, "call", func_input_validator(self.call))
        
        self._io_key_mapping = dict(
            input={i:j for i,j in zip(self.input_keys, self.input_keys_internal)},
            output={i:j for i,j in zip(self.output_keys, self.output_keys_internal)},
            output_internal={i:j for i,j in zip(self.output_keys_internal, self.output_keys)},
        )
        self.graph.add_node(
            self._id,
            name=self.name,
            _obj=weakref.ref(self),
        )
           
    def __or__(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        return BaseNodeDisposable(node=self, keys=keys)
    
    def _validate(self):
        if not self.is_start_node:
            assert len(self._input_keys_binded) == len(self.input_keys_internal), \
                AssertionError

        assert (self.graph.in_degree[self._id] == 0 and self.is_start_node) or \
            (self.graph.in_degree[self._id] != 0 and not self.is_start_node), \
            f"{self.class_name} -> Node in-degree is wrong"
        assert (self.graph.out_degree[self._id] == 0 and self.is_end_node) or \
            (self.graph.out_degree[self._id] != 0 and not self.is_end_node), \
            f"{self.class_name} -> Node out-degree is wrong"
    
    async def emit(self, 
        output_value_internal: Dict,
        session_id: str,
        save_local_output: bool = False,
        wait_for_result: bool = False,
        *args, **kwargs,
    ):
        assert all(output_value_internal.values()) and set(output_value_internal.keys()) == set(self.output_keys_internal), \
            CancelledError("Emit values are wrong.")
        
        try:
            task_list = []
            accumulative_condition = False
            
            for _nod_id in self._downstream_nodes:
                # get node
                _node: BaseNode = self.graph.nodes[_nod_id]["_obj"]()
                
                mapping_dt = self.graph.edges[self._id, _nod_id]
                _input_keys_internal, _output_keys_internal, _conditional_func = \
                    mapping_dt["_input_keys"], mapping_dt["_output_keys"], mapping_dt["_conditional_func"]
                
                _condition: bool = True
                if _conditional_func:
                    _condition = _conditional_func(**output_value_internal)

                accumulative_condition = accumulative_condition or _condition
                if _condition:
                    # emit to children
                    if _node.input_object_store.get(session_id, None) is None:
                        _node.input_object_store[session_id] = {
                            _key: {} for _key in _node.input_keys_internal
                        }
                    _input_values_internal = {
                        i: output_value_internal[o] for i, o in zip(_input_keys_internal, _output_keys_internal)
                    }
                    for _k, _v in _input_values_internal.items():
                        _node.input_object_store[session_id][_k][self._id] = _v

                    task_list.append(
                        create_task(_node._event_on_execution(session_id=session_id, save_local_output=save_local_output))
                    )
                    
            if not accumulative_condition:
                raise Exception(f"Dead on emmision: {self.class_name}")
        
            if wait_for_result:
                for coro in asyncio.as_completed(task_list):
                    try: 
                        _ = await coro
                    except Exception as err:
                        raise err

        except Exception as err:
            raise CancelledError(err)

    @abstractmethod
    async def call(self, *args, **kwargs) -> BaseModel:
        raise NotImplementedError
    
    async def _execute(self, *args, **kwargs):
        assert set(kwargs.keys()) == set(self.input_keys_internal), \
            f'{self.class_name}:{self.name} -> Input parameters {list(kwargs.keys())} and internal keys {list(self.input_keys_internal)} must be the same'
        
        try:
            _output_values: Union[BaseModel, Awaitable] = self.call(**kwargs)
            if isawaitable(_output_values):
                _output_values: BaseModel = await _output_values
            return _output_values.dict()
            
        except Exception as err:
            raise err
    
            
    def _get_local_store(self, session_id: str) -> Dict:
        dt_local_store = {}
        for _key, _node_id_list in self._input_keys_nodes_map.items():
            for _node_id in _node_id_list:
                _val = self.input_object_store.get(session_id, {}).get(_key, {}).get(_node_id, None)
                if _val:
                    dt_local_store.update({
                        _key: _val
                    })
                    break # TODO: only fetch the first valid value if no conditional function
        
        return dt_local_store

    async def _event_on_execution(self,
        session_id: str,
        save_local_output: bool = False,
        *args, **kwargs
    ):
        try:
            _inputs: Dict = self._get_local_store(session_id)
            if not set(_inputs.keys()).issubset(self.input_keys_internal) or \
                not set(self.NodeDataInput.schema()["required"]).issubset(_inputs.keys()):
                warnings.warn(f"{self.class_name}:{self.name} -> Not called")
                return None
            
            self.input_object_store.pop(session_id, None)
            
            _outputs: Dict = await self._execute(**_inputs)
            if _outputs:
                if save_local_output:
                    self.output_object_store[session_id] = {
                        self._io_key_mapping["output_internal"][k]: v for k, v in _outputs.items()
                    }

                if not self.is_end_node:
                    _ = await self.emit(
                        output_value_internal=_outputs,
                        session_id=session_id,
                        save_local_output=save_local_output
                    )
            elif self.output_keys_internal:
                raise Exception(f"{self.class_name}:{self.name} -> Output values cannot be None")
            else:
                pass
                
            return _outputs
            
        except Exception as err:
            raise err

    async def trigger(self, 
        session_id: str,
        inputs: Dict,
        save_local_output: bool = False,
        wait_for_resule: bool = True,
        *args, **kwargs
    ):
        try:
            inputs = {self._io_key_mapping["input"][k]: v for k, v in inputs.items()}
            _outputs: Dict = await self._execute(**inputs)
            
            if _outputs:
                if save_local_output:
                    self.output_object_store[session_id] = {
                        self._io_key_mapping["output_internal"][k]: v for k, v in _outputs.items()
                    }
            
                if not self.is_end_node:
                    _ = await self.emit(
                        output_value_internal=_outputs,
                        session_id=session_id,
                        save_local_output=save_local_output,
                        wait_for_result=wait_for_resule
                    )
            elif self.output_keys_internal:
                raise Exception(f"{self.class_name}:{self.name} -> Output values cannot be None")
            else:
                pass
                raise NotImplementedError
            
            return _outputs
            
        except Exception as err:
            raise err
    
    def _make_fire(self):
        for _node_name in self.graph.nodes:
            _node = self.graph.nodes[_node_name]["_obj"]()
            object.__setattr__(_node, "fire", MethodType(fire_single, _node))
        return self
          

class NodeGroup(List[str]):
    """"""
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.__repr_str__(", ")})'

    def __repr_str__(self, join_str: str):
        return join_str.join(self.__iter__())
    