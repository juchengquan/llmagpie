from __future__ import annotations
from inspect import getfullargspec
from pydantic import BaseModel
from deprecated import deprecated
from logging import Logger

from llmagpie.core.logging import get_or_create_logger

# typing
from typing import List, Sequence, Dict, Callable, Union, Any, Optional


class BaseNodeDisposable(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    connectable: Any
    in_keys: List[str] = []
    out_keys: List[str] = []

    logger: Logger

    def __init__(self, *args, **kwargs):
        logger = get_or_create_logger(self.__class__.__name__)
        super().__init__(logger=logger, *args, **kwargs)

    def _check_keys_subset(self, _keys_mapped: list, _keys_full_list: list):
        try:
            assert set(_keys_mapped).issubset(_keys_full_list)
        except AssertionError as exc:
            self.logger.error(str(exc) + f'{_keys_mapped};{_keys_full_list}')
            raise AssertionError(str(exc) + f'{_keys_mapped};{_keys_full_list}')

    def _check_if_key_in_subset(self, _key: str, _keys_full_list: list):
        assert _key in _key, AssertionError(f'{_key} not in {_keys_full_list}')


    def _check_keys_intersection(self, _keys_mapped: list, _keys_list: set):
        try:
            assert set(_keys_mapped).intersection(_keys_list) == set()
        except AssertionError as exc:
            self.logger.error(str(exc) + f'{_keys_mapped};{_keys_list}')
            raise AssertionError(str(exc) + f'{_keys_mapped};{_keys_list}')

    def __lshift__(self, node_runnables: "BaseNodeDisposable"):
        self._set_edge(node_runnables, upstream=True)
        return node_runnables

    def __rshift__(self, node_runnables: "BaseNodeDisposable"):
        self._set_edge(node_runnables, upstream=False)
        return node_runnables

    def __rrshift__(self, node_runnables: "BaseNodeDisposable"):
        """Implement [Node] >> Node because list don't have __rshift__ operators.
        Note that self refer to Node.
        """
        self.__lshift__(node_runnables)
        return self

    def __rlshift__(self, node_runnables: "BaseNodeDisposable"):
        """Implement [BaseNodeDisposable] << node because list don't have __lshift__ operators.
        Note that self refer to Node.
        """
        self.__rshift__(node_runnables)
        return self

    def _set_edge(
        self,
        node_runnables: Union["BaseNodeDisposable", List["BaseNodeDisposable"]],
        upstream: bool = True,
        **kwargs
        ):
        if not isinstance(node_runnables, Sequence):
            node_runnables = [node_runnables]

        for ele_node in node_runnables:
            if upstream:
                upper, lower = ele_node, self
            else:
                upper, lower = self, ele_node
            assert len(upper.out_keys) == len(lower.in_keys), "The key mapping must be the same!"
            
            _from_node_to_node(upper, lower)


def _from_node_to_node(upper: BaseNodeDisposable, lower: BaseNodeDisposable):
    o_node = upper.connectable
    i_node = lower.connectable
    
    o_schema = o_node._output_schema_all["internal"]
    i_schema = i_node._input_schema_all["internal"]

    _in_keys, _out_keys = [], []
    for i_key, o_key in zip(lower.in_keys, upper.out_keys):
        o_key_schema = o_schema[o_key].get("type", "object")
        i_key_schema = i_schema[i_key].get("type", "object")
        assert i_key_schema == o_key_schema, f'The schema does not align: input: {i_key}->{i_key_schema}; output: {o_key}->{o_key_schema}'
        # TODO: CHECK keys
        upper._check_if_key_in_subset(i_key, o_schema)
        lower._check_if_key_in_subset(o_key, i_schema)

        # bind all output keys to input
        i_node._input_keys_nodes_map[i_key] = i_node._input_keys_nodes_map.get(i_key, [])
        i_node._input_keys_nodes_map[i_key].append(o_node._id)

        i_node._input_keys_binded.add(i_key)

        _in_keys.append(i_key)
        _out_keys.append(o_key)
    
    i_node.is_start = False
    o_node.is_end = False

    assert all(ele in o_node.pipeline.graph for ele in [o_node._id, i_node._id])

    if o_node.pipeline.graph.has_edge(o_node._id, i_node._id):
        edge_data=o_node.pipeline.graph.get_edge_data(o_node._id, i_node._id)
        for key in o_node.pipeline.graph[o_node._id][i_node._id]:
            if key in edge_data:
                if key == '_output_keys':
                    o_node.pipeline.graph[o_node._id][i_node._id][key] += upper.out_keys
                if key == '_input_keys':
                    o_node.pipeline.graph[o_node._id][i_node._id][key] += lower.in_keys
    else:
        o_node.pipeline.graph.add_edge(
            u_of_edge=o_node._id,
            v_of_edge=i_node._id,
            ##
            _output_keys=_out_keys,
            _input_keys=_in_keys
        )
