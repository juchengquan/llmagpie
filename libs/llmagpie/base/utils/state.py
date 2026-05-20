from enum import Enum

from llmagpie.base.connectable import BaseConnectable
from pydantic import BaseModel, ConfigDict


class StateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    timestamp: float
    type: Enum
    value: dict
    node: BaseConnectable

    def to_dict(self, recursive: bool = False):
        if recursive:
            return self.model_dump()
        else:
            return self.__dict__
