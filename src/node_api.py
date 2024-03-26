from abc import abstractmethod
from ._base import Node

class NodeAPI(Node):
    # TODO
    api_route: str = ""
    url: str = ""

class NodeAPICallback(NodeAPI):
    # TODO
    url_cb: str = ""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # @abstractmethod
    async def api(self):
        """"""
    
    async def api_callback(self):
        """"""
        
    async def execute(self, session_id: str, *args, **kwargs):
        """"""
        # return await super().execute(session_id, *args, **kwargs)
    