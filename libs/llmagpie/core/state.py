from typing import List, Dict
from abc import abstractmethod, ABC


class BaseState(ABC):
    @abstractmethod    
    def clear(self):
        raise NotImplementedError
    
class ListState(List, BaseState):
    ...

class DictState(Dict, BaseState):
    ...

class InternalDictState(Dict, BaseState):
    ...
