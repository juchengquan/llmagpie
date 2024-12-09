from typing import List, Dict

class BaseState:
    def __init__(self) -> None:
        ...
    
class ListState(BaseState, List):
    ...

class DictState(BaseState, Dict):
    ...

class InternalDictState(BaseState, Dict):
    ...
