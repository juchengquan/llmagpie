from __future__ import annotations
from pydantic import BaseModel
from logging import Logger

from llmagpie.core.logging import get_or_create_logger

# typing
from typing import List, Any # Sequence, Dict, Callable, Union, Optional


class BaseConnectDisposable(BaseModel):
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

    # def _check_keys_subset(self, _keys_mapped: list, _keys_full_list: list):
    #     try:
    #         assert set(_keys_mapped).issubset(_keys_full_list)
    #     except AssertionError as exc:
    #         self.logger.error(str(exc) + f'{_keys_mapped};{_keys_full_list}')
    #         raise AssertionError(str(exc) + f'{_keys_mapped};{_keys_full_list}')

    # def _check_if_key_in_subset(self, _key: str, _keys_full_list: list):
    #     assert _key in _keys_full_list, AssertionError(f'{_key} not in {_keys_full_list}')

    # def _check_keys_intersection(self, _keys_mapped: list, _keys_list: set):
    #     try:
    #         assert set(_keys_mapped).intersection(_keys_list) == set()
    #     except AssertionError as exc:
    #         self.logger.error(str(exc) + f'{_keys_mapped};{_keys_list}')
    #         raise AssertionError(str(exc) + f'{_keys_mapped};{_keys_list}')

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
