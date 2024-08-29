import uuid
from inspect import getfullargspec, signature, _empty
from typing import Callable, Dict, Any, Union, Optional
from pydantic import BaseModel, Field, create_model
from pydantic._internal._model_construction import ModelMetaclass
from functools import wraps

def create_schema_from_function(
    function: Callable,
    name: Optional[str] = None,
    in_class: bool = False
):
    if not name:
        name = function.__name__
    args = getfullargspec(function)
    if args.varargs or args.varkw:
        raise ValueError("arg or kwargs is now allowed in function definition...")
    
    parameters = signature(function).parameters
    fields: Dict = {}
    for (idx, p_name), p_val in zip(enumerate(parameters.keys()), parameters.values()):
        if in_class and idx == 0:
            continue
        p_type = p_val.annotation
        p_default = p_val.default
        
        if p_type == _empty:
            p_type = Any
        
        if p_default == _empty:
            p_default = Field()
        else: 
            p_default = Field(default=p_default)
        
        fields[p_name] = (p_type, p_default)
    return create_model(name, __base__=BaseModel, **fields)

def func_input_validator(func):
    args = getfullargspec(func)
    if args.varargs or args.varkw:
        raise ValueError("arg or kwargs is now allowed in function definition.")
        
    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(func, "__self__"):
            Schema = create_schema_from_function(func.__func__, in_class=True)
        else:
            Schema = create_schema_from_function(func, in_class=False)
        
        return func(**Schema(**kwargs).model_dump())
        
        
    return wrapper


async def fire_single(
    cls_object,
    inputs: Union[Dict, BaseModel, ModelMetaclass],
    session_id: Optional[str] = None,
    save_local_output: bool = True,
    wait_for_result: bool = True,
):   
    res, error = None, None
    try:
        session_id = uuid.uuid4().hex if not session_id else session_id
        _node_name = cls_object.graph.root_nodes[0]
        root_node = cls_object.graph.nodes[_node_name]["_obj"]()
        
        await root_node._event_on_execution(
            session_id,
            inputs,
            wait_for_result=wait_for_result,
        )

    except Exception as err:
        error = err
    finally:
        res = {}
        for _node_id in cls_object.graph.nodes:
            _node = cls_object.graph.nodes[_node_id]["_obj"]()
            if not save_local_output:
                if _node.is_end_node:
                    res[_node.name] = _node.output_object_store.pop(session_id, None)
                else:
                    _node.output_object_store.pop(session_id, None)
            else:
                res[_node.name] = _node.output_object_store.pop(session_id, None)

        return res, error