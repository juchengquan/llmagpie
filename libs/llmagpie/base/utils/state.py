from typing import List, Dict, Optional
from pydantic import BaseModel, PrivateAttr
from abc import abstractmethod, ABC
from llmagpie.base.connectable import BaseConnectable
from enum import Enum

class StateResponse(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        
    timestamp: float
    type: Enum
    value: Dict
    node: BaseConnectable
    
    def to_dict(self, recursive: bool = False):
        if recursive:
            return self.model_dump()
        else:
            return self.__dict__
