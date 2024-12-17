from pydantic import BaseModel, PrivateAttr
from llmagpie.core.connectable import BaseConnectable
from typing import Any, Dict, Optional


class StateInput(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

class StateResponse(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        
    timestamp: float
    type: Optional[str] = None
    value: Dict
    node: BaseConnectable
    
    def to_dict(self, recursive: bool = False):
        if recursive:
            return self.model_dump()
        else:
            return self.__dict__
