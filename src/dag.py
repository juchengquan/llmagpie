from abc import ABC, abstractmethod
from networkx import DiGraph
from fastapi import APIRouter

class _Graph(DiGraph, ABC):
    def __repr__(self):
        return f"'{self.__str__()}'"
    # id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    
    @abstractmethod
    def validate_head(self, *args, **kwargs):
        """"""
        
    # @abstractmethod
    # def validate_tail(self, *args, **kwargs):
    #     """"""
    
class GraphWithAPI(_Graph):
    def validate_head(self, *args, **kwargs): # router: APIRouter, router_cb: APIRouter
        try:
            assert all(_key in kwargs for _key in ["router", "router_cb"])
            router = kwargs["router"]
            router_cb = kwargs["router_cb"]
            assert isinstance(router, APIRouter)
            assert isinstance(router_cb, APIRouter)
        except AssertionError as err:
            raise err
        
        try:
            for n, degree in self.in_degree():
                if degree == 0:
                    node = self.nodes[n]["object"]
                    assert(hasattr(node, "api"))
                    assert(hasattr(node, "api_callback"))
                    
                    router.add_api_route(
                        path=node.api_route,
                        endpoint=node.api,
                        methods=["POST"],
                    )
                    router_cb.add_api_route(
                        path=node.api_route,
                        endpoint=node.api_callback,
                        methods=["POST"]
                    )
        except AssertionError as err:
            raise err
        except Exception as err:
            raise err
        """"""

# global dag
# TODO
# try:
DAG = GraphWithAPI()
# except:
#     DAG = None